import torch
from einops import rearrange
from torch import nn


TEMPORAL_INPUT_BASES = frozenset({'absolute', 'causal_transition'})


def causal_transition_basis(emb: torch.Tensor) -> torch.Tensor:
    """Return a prefix-invertible temporal basis without future leakage.

    ``[z_0, z_1, ..., z_{T-1}]`` becomes
    ``[z_0, z_1-z_0, ..., z_{T-1}-z_{T-2}]``.  Every output prefix depends
    only on the corresponding input prefix, and the absolute trajectory is
    recovered exactly with a cumulative sum.
    """

    if emb.ndim != 3:
        raise ValueError(
            'LeWM temporal input must have shape (B,T,D), '
            f'got {tuple(emb.shape)}'
        )
    if emb.size(1) < 1:
        raise ValueError('LeWM temporal input must contain at least one token')
    return torch.cat([emb[:, :1], emb[:, 1:] - emb[:, :-1]], dim=1)


class LeWM(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        temporal_input_basis: str = 'absolute',
        **kwargs,
    ):
        super().__init__()

        temporal_input_basis = str(temporal_input_basis).strip().lower()
        if temporal_input_basis not in TEMPORAL_INPUT_BASES:
            supported = ', '.join(sorted(TEMPORAL_INPUT_BASES))
            raise ValueError(
                f'Unsupported temporal input basis {temporal_input_basis!r}; '
                f'expected one of: {supported}'
            )

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        # A string changes no state-dict entry or trainable parameter.  The
        # explicit attribute makes the inference transform travel through the
        # saved Hydra config instead of living only in a training wrapper.
        self.temporal_input_basis = temporal_input_basis

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        b = pixels.size(0)
        pixels = rearrange(
            pixels, 'b t ... -> (b t) ...'
        )  # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=b)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        temporal_input_basis = getattr(
            self, 'temporal_input_basis', 'absolute'
        )
        if temporal_input_basis == 'causal_transition':
            predictor_input = causal_transition_basis(emb)
        elif temporal_input_basis == 'absolute':
            predictor_input = emb
        else:
            raise ValueError(
                'Loaded LeWM checkpoint has unsupported temporal input basis '
                f'{temporal_input_basis!r}'
            )
        preds = self.predictor(predictor_input, act_emb)
        preds = self.pred_proj(rearrange(preds, 'b t d -> (b t) d'))
        preds = rearrange(preds, '(b t) d -> b t d', b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = None):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, H, C, h, w) — H context frames (block timesteps)
        action_sequence: (B, S, T, action_dim) — strictly-future candidates
        info['action_history']: (B, S, H - 1, action_dim) — executed action
            blocks between the context frames (required when H > 1)
         - S is the number of action plan samples
         - T is the planning horizon
        Returns ``info`` with ``predicted_emb`` of shape (B, S, H + T, D);
        the first H entries are the encoded context frames.
        """
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)

        assert 'pixels' in info, 'pixels not in info_dict'
        H = info['pixels'].size(2)
        B, S, T = action_sequence.shape[:3]
        act_past = info.get('action_history')
        if act_past is None:
            act_past = action_sequence.new_zeros(
                B, S, 0, action_sequence.size(-1)
            )
        assert act_past.size(2) == H - 1, (
            f'action_history must hold H-1={H - 1} executed blocks, '
            f'got {act_past.size(2)}'
        )
        # action paired with context frame k is the block leaving it; the
        # current frame (k = H-1) pairs with the first candidate
        info['action'] = torch.cat(
            [act_past, action_sequence[:, :, :1]], dim=2
        )

        # encode initial state, or reuse cached embedding from a prior rollout.
        # detach: to avoid backprop in encoder
        if 'emb' not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            info['emb'] = (
                _init['emb'].detach().unsqueeze(1).expand(B, S, -1, -1)
            )

        # flatten batch and sample dimensions for rollout
        emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...')
        act_past_flat = rearrange(act_past, 'b s ... -> (b s) ...')
        act_cand_flat = rearrange(action_sequence, 'b s ... -> (b s) ...')
        all_act_emb = self.action_encoder(
            torch.cat([act_past_flat, act_cand_flat], dim=1)
        )  # (BS, H - 1 + T, A_emb); index k = block leaving frame k

        # rollout predictor autoregressively, one step per candidate
        # emb_list holds individual (BS, D) frames, each with its own grad_fn
        HS = history_size
        emb_list = list(emb_init.unbind(dim=1))  # H tensors of shape (BS, D)
        for t in range(T):
            lo = max(0, H + t - HS)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)  # (BS, HS, D)
            act_trunc = all_act_emb[:, lo : H + t]  # (BS, HS, A_emb)
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        emb = torch.stack(emb_list, dim=1)  # (BS, H + T, D)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, '(b s) ... -> b s ...', b=B, s=S)
        info['predicted_emb'] = pred_rollout

        return info


__all__ = ['LeWM', 'TEMPORAL_INPUT_BASES', 'causal_transition_basis']
