import math

import torch
import torch.nn.functional as F
from einops import einsum


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Warning: This version only support single-gpu.
    Reference: https://arxiv.org/abs/2511.08544

    Args:
        knots: Number of characteristic-function integration knots.
        num_proj: Number of random feature projections.
        tail_fraction: Fraction of the largest per-projection statistics to
            average at every time index.  The default ``1.0`` is exactly the
            native mean reduction.  Values below one implement a projection
            tail-risk objective without changing the SIGReg statistic or its
            standard-Gaussian target.
        overdispersion_weight: Relative weight on the negative real
            characteristic-function residual.  For a centered Gaussian
            projection, positive residuals indicate scales below one and
            negative residuals indicate scales above one.  ``1.0`` is the
            exact native symmetric statistic; values below one prioritize
            resistance to underdispersion while retaining the imaginary
            residual's centering/asymmetry pressure.
    """

    def __init__(
        self,
        knots=17,
        num_proj=1024,
        tail_fraction=1.0,
        overdispersion_weight=1.0,
    ):
        super().__init__()
        if (
            not math.isfinite(tail_fraction)
            or tail_fraction <= 0
            or tail_fraction > 1
        ):
            raise ValueError("tail_fraction must be in the interval (0, 1]")
        if (
            not math.isfinite(overdispersion_weight)
            or overdispersion_weight < 0
            or overdispersion_weight > 1
        ):
            raise ValueError(
                "overdispersion_weight must be in the interval [0, 1]"
            )
        self.num_proj = num_proj
        self.tail_fraction = float(tail_fraction)
        self.overdispersion_weight = float(overdispersion_weight)
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer('t', t)
        self.register_buffer('phi', window)
        self.register_buffer('weights', weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        statistic = self._projected_statistic(proj, A)
        return self._aggregate_projected_statistic(statistic)

    def _aggregate_projected_statistic(self, statistic):
        """Reduce projection statistics, optionally using their upper tail."""

        if self.tail_fraction == 1.0:
            return statistic.mean()  # exact native reduction
        tail_count = max(
            1,
            math.ceil(self.tail_fraction * statistic.size(-1)),
        )
        tail = torch.topk(
            statistic,
            k=tail_count,
            dim=-1,
            largest=True,
            sorted=False,
        ).values
        return tail.mean()

    def _projected_statistic(self, proj, projections):
        """Return the Epps-Pulley statistic for each leading slice."""

        # compute the epps-pulley statistic
        x_t = (proj @ projections).unsqueeze(-1) * self.t
        cosine_residual = x_t.cos().mean(-3) - self.phi
        if self.overdispersion_weight == 1.0:
            cosine_error = cosine_residual.square()
        else:
            cosine_error = F.relu(cosine_residual).square()
            if self.overdispersion_weight > 0:
                cosine_error = cosine_error + (
                    self.overdispersion_weight
                    * F.relu(-cosine_residual).square()
                )
        err = cosine_error + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic


class TaggedAnchoredJointCFSIGReg(SIGReg):
    """Native SIGReg anchor plus a terminal conditional joint CF sketch.

    The anchor is native :class:`SIGReg` on every encoder time slice.  The
    joint component compares the empirical characteristic functions of
    ``(C, Y)`` and ``(C, Y_hat)`` at the terminal transition, where ``C`` is
    the direct concatenation of all causal history embeddings and all raw
    actions.  Condition and outcome projection directions are normalized as
    separate blocks and combined with equal ``1 / sqrt(2)`` scale.  The real
    and predicted branches always share the same projected condition and
    frequencies.

    This module has no learned parameters, consumes no pair metadata, and
    does not detach any input.  Component tensors are returned on demand and
    are never cached on the module.

    Args:
        knots: Number of characteristic-function integration knots.
        num_proj: Number of random feature projections.
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__(knots=knots, num_proj=num_proj)

    @staticmethod
    def _validate_inputs(embeddings, actions, predictions, generator):
        if not all(
            isinstance(value, torch.Tensor)
            for value in (embeddings, actions, predictions)
        ):
            raise TypeError(
                "embeddings, actions, and predictions must be tensors"
            )
        if embeddings.dim() != 3:
            raise ValueError(
                "embeddings must have shape (B, H + 1, D), "
                f"got {tuple(embeddings.shape)}"
            )
        if actions.dim() != 3:
            raise ValueError(
                "actions must have shape (B, H, A), "
                f"got {tuple(actions.shape)}"
            )
        if predictions.dim() != 3:
            raise ValueError(
                "predictions must have shape (B, H, D), "
                f"got {tuple(predictions.shape)}"
            )

        batch_size, sequence_length, embedding_dim = embeddings.shape
        horizon = actions.size(1)
        if batch_size < 1 or horizon < 1:
            raise ValueError(
                "batch size and prediction horizon must be positive"
            )
        if embedding_dim < 1 or actions.size(-1) < 1:
            raise ValueError(
                "embedding and action dimensions must be positive"
            )
        if sequence_length != horizon + 1:
            raise ValueError(
                "embeddings must contain H causal slices and one terminal "
                f"target: got S={sequence_length} and H={horizon}"
            )
        if actions.size(0) != batch_size:
            raise ValueError("actions and embeddings must share batch size")
        expected_prediction_shape = (batch_size, horizon, embedding_dim)
        if tuple(predictions.shape) != expected_prediction_shape:
            raise ValueError(
                "predictions must have shape (B, H, D), "
                f"expected {expected_prediction_shape}, got "
                f"{tuple(predictions.shape)}"
            )
        devices = {embeddings.device, actions.device, predictions.device}
        if len(devices) != 1:
            raise ValueError(
                "embeddings, actions, and predictions must share a device"
            )
        if not all(
            torch.is_floating_point(value)
            for value in (embeddings, actions, predictions)
        ):
            raise TypeError(
                "embeddings, actions, and predictions must be floating point"
            )
        if generator is not None and not isinstance(
            generator, torch.Generator
        ):
            raise TypeError("generator must be a torch.Generator or None")

    def _sample_unit_projections(self, dimension, device, generator):
        projections = torch.randn(
            dimension,
            self.num_proj,
            device=device,
            generator=generator,
        )
        return projections.div_(projections.norm(p=2, dim=0))

    def _anchor_component(self, embeddings, generator):
        native_layout = embeddings.transpose(0, 1)
        if generator is None:
            # Preserve the native call path as well as its RNG and reduction
            # order so the anchor is bit-exact under a shared global seed.
            return SIGReg.forward(self, native_layout)

        projections = self._sample_unit_projections(
            embeddings.size(-1),
            embeddings.device,
            generator,
        )
        statistic = self._projected_statistic(
            native_layout,
            projections,
        )
        # This candidate freezes the native mean reduction.  Keeping the
        # reduction explicit also makes the class directly overlay-compatible
        # with the benchmark's pinned pre-extension SIGReg implementation.
        return statistic.mean()

    def _joint_component(
        self,
        condition,
        target,
        prediction,
        generator,
    ):
        condition_projections = self._sample_unit_projections(
            condition.size(-1),
            condition.device,
            generator,
        )
        outcome_projections = self._sample_unit_projections(
            target.size(-1),
            target.device,
            generator,
        )
        block_projections = torch.cat(
            [condition_projections, outcome_projections],
            dim=0,
        ) / math.sqrt(2.0)

        real_joint = torch.cat([condition, target], dim=-1)
        predicted_joint = torch.cat([condition, prediction], dim=-1)
        real_arguments = (
            (real_joint @ block_projections).unsqueeze(-1) * self.t
        )
        predicted_arguments = (
            (predicted_joint @ block_projections).unsqueeze(-1) * self.t
        )
        cosine_difference = (
            real_arguments.cos().mean(dim=-3)
            - predicted_arguments.cos().mean(dim=-3)
        )
        sine_difference = (
            real_arguments.sin().mean(dim=-3)
            - predicted_arguments.sin().mean(dim=-3)
        )
        statistic = (
            (cosine_difference.square() + sine_difference.square())
            @ self.weights
        ) * condition.size(0)
        return statistic.mean()

    def forward_with_components(
        self,
        embeddings,
        actions,
        predictions,
        *,
        generator=None,
    ):
        """Return freshly computed ``total``, ``anchor``, and ``joint``.

        Args:
            embeddings: Encoder embeddings with shape ``(B, H + 1, D)``.
            actions: Raw normalized actions with shape ``(B, H, A)``.
            predictions: Predictor outputs with shape ``(B, H, D)``.
            generator: Optional same-device generator controlling, in order,
                anchor, condition-block, and outcome-block projections.
        """

        self._validate_inputs(embeddings, actions, predictions, generator)
        horizon = actions.size(1)
        anchor = self._anchor_component(embeddings, generator)
        condition = torch.cat(
            [
                embeddings[:, :horizon].reshape(embeddings.size(0), -1),
                actions.reshape(actions.size(0), -1),
            ],
            dim=-1,
        )
        joint = self._joint_component(
            condition,
            embeddings[:, horizon],
            predictions[:, -1],
            generator,
        )
        total = anchor + joint
        return {
            'total': total,
            'anchor': anchor,
            'joint': joint,
        }

    def components(
        self,
        embeddings,
        actions,
        predictions,
        *,
        generator=None,
    ):
        """Alias for :meth:`forward_with_components` for audit callers."""

        return self.forward_with_components(
            embeddings,
            actions,
            predictions,
            generator=generator,
        )

    def forward(
        self,
        embeddings,
        actions,
        predictions,
        *,
        generator=None,
    ):
        """Return the scalar tagged anchor-plus-joint regularizer."""

        return self.forward_with_components(
            embeddings,
            actions,
            predictions,
            generator=generator,
        )['total']


class MixedRadiusTaggedAnchoredJointCFSIGReg(
    TaggedAnchoredJointCFSIGReg
):
    """Tagged anchor plus a mixed-radius terminal joint-CF sketch.

    This candidate preserves :class:`TaggedAnchoredJointCFSIGReg` except for
    the joint-frequency sampling law.  For every projection it samples unit
    condition and outcome directions, a uniform block-energy fraction ``u``,
    and a half-normal total radius ``rho``.  The concatenated frequency is

    ``rho * [sqrt(u) * v_condition, sqrt(1-u) * v_outcome]``.

    The continuous population frequency law has full topological support and
    keeps expected squared energy one half in each block.  A finite realized
    projection set is only a random sketch and is not itself a conditional-law
    identification theorem.  The native anchor, inputs, condition, component
    contract, and gradient paths are inherited unchanged.
    """

    @staticmethod
    def _mixed_radius_scales(mixing, radius):
        return radius * mixing.sqrt(), radius * (1.0 - mixing).sqrt()

    def _sample_mixed_radius_block_projections(
        self,
        condition_dimension,
        outcome_dimension,
        device,
        generator,
    ):
        condition_projections = self._sample_unit_projections(
            condition_dimension,
            device,
            generator,
        )
        outcome_projections = self._sample_unit_projections(
            outcome_dimension,
            device,
            generator,
        )
        projection_dtype = condition_projections.dtype
        mixing = torch.rand(
            self.num_proj,
            device=device,
            dtype=projection_dtype,
            generator=generator,
        )
        radius = torch.randn(
            self.num_proj,
            device=device,
            dtype=projection_dtype,
            generator=generator,
        ).abs()
        condition_scale, outcome_scale = self._mixed_radius_scales(
            mixing,
            radius,
        )
        return torch.cat(
            [
                condition_projections * condition_scale.unsqueeze(0),
                outcome_projections * outcome_scale.unsqueeze(0),
            ],
            dim=0,
        )

    def _joint_component(
        self,
        condition,
        target,
        prediction,
        generator,
    ):
        block_projections = self._sample_mixed_radius_block_projections(
            condition.size(-1),
            target.size(-1),
            condition.device,
            generator,
        )

        real_joint = torch.cat([condition, target], dim=-1)
        predicted_joint = torch.cat([condition, prediction], dim=-1)
        real_arguments = (
            (real_joint @ block_projections).unsqueeze(-1) * self.t
        )
        predicted_arguments = (
            (predicted_joint @ block_projections).unsqueeze(-1) * self.t
        )
        cosine_difference = (
            real_arguments.cos().mean(dim=-3)
            - predicted_arguments.cos().mean(dim=-3)
        )
        sine_difference = (
            real_arguments.sin().mean(dim=-3)
            - predicted_arguments.sin().mean(dim=-3)
        )
        statistic = (
            (cosine_difference.square() + sine_difference.square())
            @ self.weights
        ) * condition.size(0)
        return statistic.mean()


class _IEEEFP32Projection(torch.autograd.Function):
    """FP32 projection with TF32 disabled in forward and input backward."""

    @staticmethod
    def _assert_device_predicate(predicate, message):
        try:
            torch._assert_async(predicate, message)
        except RuntimeError as error:
            if predicate.device.type == "cpu":
                raise FloatingPointError(message) from error
            raise

    @staticmethod
    def _validate_forward_inputs(x, direction):
        if not isinstance(x, torch.Tensor) or not isinstance(
            direction, torch.Tensor
        ):
            raise TypeError("x and direction must be tensors")
        if x.dtype != torch.float32 or direction.dtype != torch.float32:
            raise TypeError("x and direction must have dtype torch.float32")
        if x.device != direction.device:
            raise ValueError("x and direction must share a device")
        if x.dim() != 2 or direction.dim() != 2:
            raise ValueError("x and direction must both be rank-two tensors")
        if x.size(1) != direction.size(0):
            raise ValueError(
                "x and direction have incompatible projection dimensions"
            )
        if direction.requires_grad:
            raise ValueError("random directions must not require gradients")
        finite = torch.stack(
            (torch.isfinite(x).all(), torch.isfinite(direction).all())
        ).all()
        _IEEEFP32Projection._assert_device_predicate(
            finite,
            "x and direction must be finite for FP32 projection",
        )

    @staticmethod
    def _matmul_without_tf32(left, right):
        if left.device.type != "cuda":
            return torch.matmul(left, right)

        previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            return torch.matmul(left, right)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32

    @staticmethod
    def forward(ctx, x, direction):
        _IEEEFP32Projection._validate_forward_inputs(x, direction)
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = _IEEEFP32Projection._matmul_without_tf32(x, direction)
        if output.dtype != torch.float32:
            raise TypeError("FP32 projection produced a non-float32 output")
        _IEEEFP32Projection._assert_device_predicate(
            torch.isfinite(output).all(),
            "FP32 projection produced a nonfinite output",
        )
        ctx.save_for_backward(direction)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (direction,) = ctx.saved_tensors
        grad_output = grad_output.to(dtype=torch.float32)
        if grad_output.device != direction.device:
            raise ValueError(
                "projection gradient and direction must share a device"
            )
        if grad_output.dim() != 2:
            raise ValueError("projection gradient must be rank two")
        if grad_output.size(1) != direction.size(1):
            raise ValueError(
                "projection gradient and direction have incompatible shapes"
            )
        _IEEEFP32Projection._assert_device_predicate(
            torch.isfinite(grad_output).all(),
            "projection gradient must be finite",
        )
        direction_transpose = direction.transpose(0, 1).contiguous()
        with torch.autocast(
            device_type=grad_output.device.type,
            enabled=False,
        ):
            grad_x = _IEEEFP32Projection._matmul_without_tf32(
                grad_output,
                direction_transpose,
            )
        if grad_x.dtype != torch.float32:
            raise TypeError("FP32 projection backward produced a bad dtype")
        _IEEEFP32Projection._assert_device_predicate(
            torch.isfinite(grad_x).all(),
            "FP32 projection backward produced a nonfinite gradient",
        )
        return grad_x, None


class SemanticBlockTaggedAnchoredJointSWReg(TaggedAnchoredJointCFSIGReg):
    """Native SIGReg anchor plus semantic-block joint sliced Wasserstein.

    The terminal real and predicted joint samples use separate unit-sphere
    directions for causal history embeddings, normalized raw actions, and
    terminal outcomes.  Fresh Dirichlet(1, 1, 1) block energies allocate the
    projection energy symmetrically across those three semantic blocks.  The
    empirical one-dimensional transport cost sums local batch rows before it
    averages projection columns.

    The joint path is an FP32 autocast island.  Its four projection matrix
    multiplications use :class:`_IEEEFP32Projection`, which also protects the
    input-gradient matrix multiplications from CUDA TF32.  The module adds no
    learned parameters, buffers, heads, metadata, or graph caches.
    """

    @staticmethod
    def _fp32_finite_stage_predicate(stage, *tensors):
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a nonempty string")
        if not tensors:
            raise ValueError("at least one tensor is required")
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{stage} values must be tensors")
            if tensor.dtype != torch.float32:
                raise TypeError(
                    f"{stage} values must have dtype torch.float32"
                )
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            raise ValueError(f"{stage} values must share a device")
        return torch.stack(
            tuple(torch.isfinite(tensor).all() for tensor in tensors)
        ).all()

    @staticmethod
    def _require_fp32_finite_stage_and_tensors(stage, *tensors):
        predicate_builder = (
            SemanticBlockTaggedAnchoredJointSWReg._fp32_finite_stage_predicate
        )
        finite = predicate_builder(stage, *tensors)
        if not bool(finite):
            raise FloatingPointError(f"nonfinite value in {stage}")

    def _sample_fp32_unit_directions(
        self,
        dimension,
        device,
        generator,
        finite_predicates=None,
    ):
        directions = torch.randn(
            dimension,
            self.num_proj,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        norms = directions.norm(p=2, dim=0)
        norm_is_valid = self._fp32_finite_stage_predicate(
            "direction_norms",
            directions,
            norms,
        ) & norms.ne(0).all()
        _IEEEFP32Projection._assert_device_predicate(
            norm_is_valid,
            "zero or nonfinite norm in sampled direction",
        )
        directions.div_(norms)
        direction_is_finite = self._fp32_finite_stage_predicate(
            "directions",
            directions,
        )
        if finite_predicates is None:
            self._require_fp32_finite_stage_and_tensors(
                "directions",
                directions,
            )
        else:
            finite_predicates.extend((norm_is_valid, direction_is_finite))
        return directions

    def _sample_fp32_block_energies_and_scales(
        self,
        device,
        generator,
        finite_predicates=None,
    ):
        uniform_1 = torch.rand(
            self.num_proj,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        uniform_2 = torch.rand(
            self.num_proj,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        radius = uniform_1.sqrt()
        history_energy = 1.0 - radius
        action_energy = radius * (1.0 - uniform_2)
        outcome_energy = radius * uniform_2
        history_scale = history_energy.sqrt()
        action_scale = action_energy.sqrt()
        outcome_scale = outcome_energy.sqrt()
        energy_is_finite = self._fp32_finite_stage_predicate(
            "energies",
            uniform_1,
            uniform_2,
            radius,
            history_energy,
            action_energy,
            outcome_energy,
            history_scale,
            action_scale,
            outcome_scale,
        )
        if finite_predicates is None:
            self._require_fp32_finite_stage_and_tensors(
                "energies",
                uniform_1,
                uniform_2,
                radius,
                history_energy,
                action_energy,
                outcome_energy,
                history_scale,
                action_scale,
                outcome_scale,
            )
        else:
            finite_predicates.append(energy_is_finite)
        return (
            (history_energy, action_energy, outcome_energy),
            (history_scale, action_scale, outcome_scale),
        )

    def _joint_component(
        self,
        history,
        actions,
        target,
        prediction,
        generator,
    ):
        with torch.autocast(
            device_type=history.device.type,
            enabled=False,
        ):
            history = history.to(dtype=torch.float32)
            actions = actions.to(dtype=torch.float32)
            target = target.to(dtype=torch.float32)
            prediction = prediction.to(dtype=torch.float32)
            finite_predicates = []
            finite_predicates.append(
                self._fp32_finite_stage_predicate(
                    "joint_inputs",
                    history,
                    actions,
                    target,
                    prediction,
                )
            )

            history_directions = self._sample_fp32_unit_directions(
                history.size(-1),
                history.device,
                generator,
                finite_predicates,
            )
            action_directions = self._sample_fp32_unit_directions(
                actions.size(-1),
                actions.device,
                generator,
                finite_predicates,
            )
            outcome_directions = self._sample_fp32_unit_directions(
                target.size(-1),
                target.device,
                generator,
                finite_predicates,
            )
            _, scales = self._sample_fp32_block_energies_and_scales(
                history.device,
                generator,
                finite_predicates,
            )
            history_scale, action_scale, outcome_scale = scales

            projected_history = _IEEEFP32Projection.apply(
                history,
                history_directions,
            )
            projected_actions = _IEEEFP32Projection.apply(
                actions,
                action_directions,
            )
            projected_target = _IEEEFP32Projection.apply(
                target,
                outcome_directions,
            )
            projected_prediction = _IEEEFP32Projection.apply(
                prediction,
                outcome_directions,
            )
            finite_predicates.append(
                self._fp32_finite_stage_predicate(
                    "projections",
                    projected_history,
                    projected_actions,
                    projected_target,
                    projected_prediction,
                )
            )

            common = projected_history * history_scale.unsqueeze(0)
            common = common + projected_actions * action_scale.unsqueeze(0)
            real_projection = (
                common + projected_target * outcome_scale.unsqueeze(0)
            )
            predicted_projection = (
                common + projected_prediction * outcome_scale.unsqueeze(0)
            )
            finite_predicates.append(
                self._fp32_finite_stage_predicate(
                    "joint_projections",
                    common,
                    real_projection,
                    predicted_projection,
                )
            )

            sorted_real = torch.sort(
                real_projection,
                dim=0,
                descending=False,
                stable=True,
            ).values
            sorted_prediction = torch.sort(
                predicted_projection,
                dim=0,
                descending=False,
                stable=True,
            ).values
            finite_predicates.append(
                self._fp32_finite_stage_predicate(
                    "sorted",
                    sorted_real,
                    sorted_prediction,
                )
            )

            transport_squared = (sorted_real - sorted_prediction).square()
            per_projection = transport_squared.sum(dim=0)
            finite_predicates.append(
                self._fp32_finite_stage_predicate(
                    "transport_squared",
                    transport_squared,
                    per_projection,
                )
            )
            joint = per_projection.mean()
            finite_predicates.append(
                self._fp32_finite_stage_predicate(
                    "final_joint",
                    joint,
                )
            )
            all_stages_are_finite = torch.stack(
                tuple(finite_predicates)
            ).all()
            finite_sentinel = torch.where(
                all_stages_are_finite,
                torch.zeros_like(joint),
                torch.full_like(joint, float("nan")),
            )
            self._require_fp32_finite_stage_and_tensors(
                "joint_path",
                finite_sentinel,
            )
            return joint

    def forward_with_components(
        self,
        embeddings,
        actions,
        predictions,
        *,
        generator=None,
    ):
        """Return freshly computed ``total``, ``anchor``, and ``joint``."""

        self._validate_inputs(embeddings, actions, predictions, generator)
        horizon = actions.size(1)
        anchor = self._anchor_component(embeddings, generator)
        history = embeddings[:, :horizon].reshape(embeddings.size(0), -1)
        action_history = actions.reshape(actions.size(0), -1)
        joint = self._joint_component(
            history,
            action_history,
            embeddings[:, horizon],
            predictions[:, -1],
            generator,
        )
        total = anchor + joint
        return {
            'total': total,
            'anchor': anchor,
            'joint': joint,
        }


class SemanticBlockTaggedAnchoredJointTailMassSWReg(
    SemanticBlockTaggedAnchoredJointSWReg
):
    """V3 semantic-block joint SW with a fixed upper-tail mass reducer.

    All 1024 registered semantic-block directions and their local-batch
    squared transport costs are computed exactly as in
    :class:`SemanticBlockTaggedAnchoredJointSWReg`.  The only algorithmic
    change is the final projection reduction: the largest
    ``ceil(0.10 * K)`` costs are summed and divided by ``K``.  This is a tail
    mass, not a tail mean, so it cannot exceed the V3 full projection mean.

    Projection costs are ranked in FP32 with a stable descending sort.  The
    original projection index therefore supplies the deterministic tie rule.
    The class adds no learned parameters, buffers, heads, or random draws.
    """

    JOINT_TAIL_FRACTION = 0.10

    def _aggregate_joint_projection_costs(self, per_projection):
        if not isinstance(per_projection, torch.Tensor):
            raise TypeError("joint projection costs must be a tensor")
        if per_projection.dtype != torch.float32:
            raise TypeError("joint projection costs must have dtype torch.float32")
        if per_projection.dim() != 1:
            raise ValueError("joint projection costs must be rank one")
        if per_projection.numel() != self.num_proj:
            raise ValueError("joint projection cost count must equal num_proj")
        costs_are_valid = self._fp32_finite_stage_predicate(
            "projection_costs",
            per_projection,
        ) & per_projection.ge(0).all()
        _IEEEFP32Projection._assert_device_predicate(
            costs_are_valid,
            "joint projection costs must be finite and nonnegative",
        )
        ordered_costs = torch.sort(
            per_projection,
            dim=0,
            descending=True,
            stable=True,
        ).values
        tail_count = max(
            1,
            math.ceil(self.JOINT_TAIL_FRACTION * self.num_proj),
        )
        selected_costs = ordered_costs[:tail_count]
        joint = selected_costs.sum() / self.num_proj
        return joint, ordered_costs, selected_costs

    def _joint_component(
        self,
        history,
        actions,
        target,
        prediction,
        generator,
    ):
        with torch.autocast(
            device_type=history.device.type,
            enabled=False,
        ):
            history = history.to(dtype=torch.float32)
            actions = actions.to(dtype=torch.float32)
            target = target.to(dtype=torch.float32)
            prediction = prediction.to(dtype=torch.float32)
            finite_predicates = []
            finite_predicates.append(
                self._fp32_finite_stage_predicate(
                    "joint_inputs",
                    history,
                    actions,
                    target,
                    prediction,
                )
            )

            history_directions = self._sample_fp32_unit_directions(
                history.size(-1),
                history.device,
                generator,
                finite_predicates,
            )
            action_directions = self._sample_fp32_unit_directions(
                actions.size(-1),
                actions.device,
                generator,
                finite_predicates,
            )
            outcome_directions = self._sample_fp32_unit_directions(
                target.size(-1),
                target.device,
                generator,
                finite_predicates,
            )
            _, scales = self._sample_fp32_block_energies_and_scales(
                history.device,
                generator,
                finite_predicates,
            )
            history_scale, action_scale, outcome_scale = scales

            projected_history = _IEEEFP32Projection.apply(
                history,
                history_directions,
            )
            projected_actions = _IEEEFP32Projection.apply(
                actions,
                action_directions,
            )
            projected_target = _IEEEFP32Projection.apply(
                target,
                outcome_directions,
            )
            projected_prediction = _IEEEFP32Projection.apply(
                prediction,
                outcome_directions,
            )
            finite_predicates.append(
                self._fp32_finite_stage_predicate(
                    "projections",
                    projected_history,
                    projected_actions,
                    projected_target,
                    projected_prediction,
                )
            )

            common = projected_history * history_scale.unsqueeze(0)
            common = common + projected_actions * action_scale.unsqueeze(0)
            real_projection = (
                common + projected_target * outcome_scale.unsqueeze(0)
            )
            predicted_projection = (
                common + projected_prediction * outcome_scale.unsqueeze(0)
            )
            finite_predicates.append(
                self._fp32_finite_stage_predicate(
                    "joint_projections",
                    common,
                    real_projection,
                    predicted_projection,
                )
            )

            sorted_real = torch.sort(
                real_projection,
                dim=0,
                descending=False,
                stable=True,
            ).values
            sorted_prediction = torch.sort(
                predicted_projection,
                dim=0,
                descending=False,
                stable=True,
            ).values
            finite_predicates.append(
                self._fp32_finite_stage_predicate(
                    "sorted",
                    sorted_real,
                    sorted_prediction,
                )
            )

            transport_squared = (sorted_real - sorted_prediction).square()
            per_projection = transport_squared.sum(dim=0)
            joint, ordered_costs, selected_costs = (
                self._aggregate_joint_projection_costs(per_projection)
            )
            finite_predicates.append(
                self._fp32_finite_stage_predicate(
                    "transport_squared",
                    transport_squared,
                    per_projection,
                )
            )
            finite_predicates.append(
                self._fp32_finite_stage_predicate(
                    "tail_mass",
                    ordered_costs,
                    selected_costs,
                    joint,
                )
            )
            all_stages_are_finite = torch.stack(
                tuple(finite_predicates)
            ).all()
            finite_sentinel = torch.where(
                all_stages_are_finite,
                torch.zeros_like(joint),
                torch.full_like(joint, float("nan")),
            )
            self._require_fp32_finite_stage_and_tensors(
                "joint_path",
                finite_sentinel,
            )
            return joint


class TemporallyCenteredSIGReg(SIGReg):
    """SIGReg on within-trajectory residuals.

    Each trajectory receives a free location parameter equal to its temporal
    mean.  The unchanged SIGReg statistic is applied to

    ``proj[t, b] - mean_t(proj[t, b])``.

    This is a parameter-free Gaussian location-mixture target: between-
    trajectory centers are not Gaussianized, while collapse within every
    trajectory still produces a degenerate residual population.  Centering
    remains fully differentiable; no branch is detached or frozen.

    Reference: https://arxiv.org/abs/2607.26924
    """

    def forward(self, proj):
        """
        proj: (T, B, D), with T >= 2
        """
        if proj.dim() != 3:
            raise ValueError(
                "TemporallyCenteredSIGReg expects proj with shape "
                f"(T, B, D), got {tuple(proj.shape)}"
            )
        if proj.size(0) < 2:
            raise ValueError(
                "TemporallyCenteredSIGReg requires at least two time steps"
            )
        residuals = proj - proj.mean(dim=0, keepdim=True)
        return super().forward(residuals)


class JointTemporalCovarianceSIGReg(SIGReg):
    """SIGReg against a random-intercept temporal Gaussian target.

    The null model for each embedding coordinate is

    ``z_t = u + epsilon_t``, where
    ``u ~ N(0, (1 - rho) I)`` and
    ``epsilon_t ~ N(0, rho I)``.

    A deterministic Helmert transform diagonalizes its temporal covariance.
    The center mode has variance ``T - (T - 1) * rho`` and each of the
    ``T - 1`` contrast modes has variance ``rho``.  After whitening those
    modes, time and feature are flattened into one vector per trajectory and
    passed to the unchanged SIGReg statistic.  Joint random projections are
    important: applying SIGReg to the modes separately would not identify
    their cross-mode covariance.

    When ``rho`` is omitted, it is resolved from the observed sequence length
    as ``1 / T**2``.  The transform is invertible and fully differentiable;
    it consumes no pair metadata and uses no detached or frozen branch.
    """

    def __init__(
        self,
        knots=17,
        num_proj=1024,
        rho=None,
        tail_fraction=1.0,
        overdispersion_weight=1.0,
    ):
        super().__init__(
            knots=knots,
            num_proj=num_proj,
            tail_fraction=tail_fraction,
            overdispersion_weight=overdispersion_weight,
        )
        if rho is not None and (
            not math.isfinite(rho) or rho <= 0.0 or rho > 1.0
        ):
            raise ValueError("rho must be None or in the interval (0, 1]")
        self.rho = None if rho is None else float(rho)

    @staticmethod
    def _helmert_basis(length, *, device, dtype):
        """Return an orthonormal basis with the constant mode first."""

        basis = torch.zeros(length, length, device=device, dtype=dtype)
        basis[0].fill_(1.0 / math.sqrt(length))
        for row in range(1, length):
            scale = 1.0 / math.sqrt(row * (row + 1))
            basis[row, :row] = scale
            basis[row, row] = -row * scale
        return basis

    def resolved_rho(self, length):
        """Resolve the fixed or sequence-length-derived innovation ratio."""

        return self.rho if self.rho is not None else 1.0 / (length**2)

    def normalized_temporal_modes(self, proj):
        """Whiten the declared temporal covariance in a Helmert basis."""

        if proj.dim() != 3:
            raise ValueError(
                "JointTemporalCovarianceSIGReg expects proj with shape "
                f"(T, B, D), got {tuple(proj.shape)}"
            )
        length = proj.size(0)
        if length < 2:
            raise ValueError(
                "JointTemporalCovarianceSIGReg requires at least two time "
                "steps"
            )
        rho = self.resolved_rho(length)
        basis = self._helmert_basis(
            length,
            device=proj.device,
            dtype=proj.dtype,
        )
        modes = torch.einsum("st,tbd->sbd", basis, proj)
        variances = proj.new_full((length,), rho)
        variances[0] = length - (length - 1) * rho
        return modes / variances.sqrt().view(length, 1, 1)

    def forward(self, proj):
        """
        proj: (T, B, D), with T >= 2
        """

        modes = self.normalized_temporal_modes(proj)
        joint = modes.permute(1, 0, 2).reshape(
            1,
            modes.size(1),
            modes.size(0) * modes.size(2),
        )
        return super().forward(joint)


class ConditionalSIGReg(SIGReg):
    """SIGReg on condition-matched representation contrasts.

    Vanilla SIGReg constrains every time-indexed marginal population
    ``z[t]``.  That population can remain Gaussian even when two futures
    belonging to the same visible query/action condition are mapped to the
    same representation.  This module replaces selected marginal slices by
    their condition-matched Haar high-pass contrasts

    ``(z[t, i] - z[t, j]) / sqrt(2)``.

    For two independent standard Gaussian representations, this contrast is
    itself standard Gaussian.  Pair collapse instead creates a zero
    population, which the same SIGReg statistic detects directly.  The
    default deliberately tests the conditional high-pass population alone
    at active time indices.

    The implementation uses the same random feature projections,
    characteristic-function statistic, Gaussian target, and single scalar
    loss as SIGReg.  No rule or class labels are consumed: callers provide
    disjoint pairs that can be constructed from visible conditioning
    variables.  With no pairs, the implementation calls :class:`SIGReg`
    exactly.

    Args:
        knots: Number of characteristic-function integration knots.
        num_proj: Number of random feature projections.
        randomize_pair_orientation: Multiply every contrast row by an
            independent Rademacher sign.  This makes the expected objective
            invariant to the arbitrary ordering inside each pair.
        include_unpaired: At an active time index, concatenate batch rows
            that do not belong to an active pair with the selected pair
            contrasts.  Both populations have the same standard-Gaussian
            null, so this preserves marginal coverage for replay samples
            without adding a second loss component.  The default is False
            for exact backward compatibility with the original conditional
            high-pass experiments.
        complete_haar_population: Apply a complete orthonormal Haar transform
            to every selected pair and concatenate its low-pass means,
            high-pass contrasts, and all unpaired rows.  This keeps the
            active population the same size as the original batch and gives
            every row one route into the single SIGReg statistic.  It is
            mutually exclusive with ``include_unpaired``.
    """

    def __init__(
        self,
        knots=17,
        num_proj=1024,
        randomize_pair_orientation=True,
        include_unpaired=False,
        complete_haar_population=False,
    ):
        super().__init__(knots=knots, num_proj=num_proj)
        if include_unpaired and complete_haar_population:
            raise ValueError(
                "include_unpaired and complete_haar_population are "
                "mutually exclusive"
            )
        self.randomize_pair_orientation = randomize_pair_orientation
        self.include_unpaired = include_unpaired
        self.complete_haar_population = complete_haar_population

    @staticmethod
    def _validate_pairs(proj, pairs, active):
        if proj.dim() != 3:
            raise ValueError(
                "ConditionalSIGReg expects proj with shape (T, B, D), "
                f"got {tuple(proj.shape)}"
            )
        if pairs.dim() != 2 or pairs.size(-1) != 2:
            raise ValueError(
                "pairs must have shape (P, 2), "
                f"got {tuple(pairs.shape)}"
            )
        if pairs.dtype != torch.long:
            raise TypeError(f"pairs must use torch.long, got {pairs.dtype}")
        if pairs.numel() and (
            int(pairs.min()) < 0 or int(pairs.max()) >= proj.size(1)
        ):
            raise ValueError(
                "pairs contain an index outside the representation batch"
            )
        flattened = pairs.flatten()
        if torch.unique(flattened).numel() != flattened.numel():
            raise ValueError("pairs must be disjoint")
        expected_active_shape = (proj.size(0), pairs.size(0))
        if tuple(active.shape) != expected_active_shape:
            raise ValueError(
                "active must have shape (T, P), "
                f"expected {expected_active_shape}, got {tuple(active.shape)}"
            )
        if active.dtype != torch.bool:
            raise TypeError(f"active must use torch.bool, got {active.dtype}")

    def forward(self, proj, pairs=None, active=None):
        """
        Args:
            proj: Representations with shape ``(T, B, D)``.
            pairs: Disjoint condition-matched indices with shape ``(P, 2)``.
            active: Boolean mask with shape ``(T, P)``.  At a time index with
                any active pairs, SIGReg is evaluated on the selected pair
                contrasts instead of the unconditional batch marginal.
        """

        if pairs is None:
            if active is not None:
                raise ValueError("active requires pairs")
            return super().forward(proj)

        pairs = pairs.to(device=proj.device)
        if active is None:
            active = torch.ones(
                proj.size(0),
                pairs.size(0),
                dtype=torch.bool,
                device=proj.device,
            )
        else:
            active = active.to(device=proj.device)
        self._validate_pairs(proj, pairs, active)
        if pairs.numel() == 0 or not bool(active.any()):
            return super().forward(proj)

        projections = torch.randn(
            proj.size(-1), self.num_proj, device=proj.device
        )
        projections = projections.div_(projections.norm(p=2, dim=0))
        rows = []
        inverse_sqrt_two = 1.0 / math.sqrt(2.0)
        for time_index in range(proj.size(0)):
            selected = active[time_index]
            if not bool(selected.any()):
                population = proj[time_index]
            else:
                selected_pairs = pairs[selected]
                contrasts = (
                    proj[time_index, selected_pairs[:, 0]]
                    - proj[time_index, selected_pairs[:, 1]]
                ) * inverse_sqrt_two
                if self.randomize_pair_orientation:
                    signs = torch.empty(
                        contrasts.size(0),
                        1,
                        device=contrasts.device,
                        dtype=contrasts.dtype,
                    )
                    signs.bernoulli_(0.5).mul_(2.0).sub_(1.0)
                    contrasts = contrasts * signs
                if self.include_unpaired or self.complete_haar_population:
                    unpaired = torch.ones(
                        proj.size(1),
                        dtype=torch.bool,
                        device=proj.device,
                    )
                    unpaired[selected_pairs.flatten()] = False
                if self.complete_haar_population:
                    means = (
                        proj[time_index, selected_pairs[:, 0]]
                        + proj[time_index, selected_pairs[:, 1]]
                    ) * inverse_sqrt_two
                    population = torch.cat(
                        [proj[time_index, unpaired], means, contrasts],
                        dim=0,
                    )
                elif self.include_unpaired:
                    population = torch.cat(
                        [proj[time_index, unpaired], contrasts],
                        dim=0,
                    )
                else:
                    population = contrasts
            rows.append(
                self._projected_statistic(
                    population.unsqueeze(0), projections
                ).mean()
            )
        return torch.stack(rows).mean()


class GroupBalancedSIGReg(ConditionalSIGReg):
    """SIGReg with separately scored marginal and paired-difference groups.

    At an active time index, native SIGReg on the complete batch marginal
    and conditional SIGReg on the matched Haar high-pass contrasts are
    computed with the same random projections and averaged:

    ``0.5 * (SIGReg(z) + SIGReg((z_i - z_j) / sqrt(2)))``.

    Computing the statistics separately is essential.  Concatenating the
    two populations lets the larger marginal population dilute a collapsed
    conditional direction.  The group-balanced objective keeps the native
    marginal route for every sample while assigning the paired-difference
    group equal scalar weight.  At inactive time indices it is exactly
    native SIGReg.

    This remains one Gaussianity statistic and one external loss weight; it
    does not introduce separate variance, covariance, or invariance terms.
    Pair metadata identifies matched visible conditions and contains no
    hidden rule or class label.
    """

    def __init__(
        self,
        knots=17,
        num_proj=1024,
        randomize_pair_orientation=True,
    ):
        super().__init__(
            knots=knots,
            num_proj=num_proj,
            randomize_pair_orientation=randomize_pair_orientation,
        )

    def forward(self, proj, pairs=None, active=None):
        """
        Args:
            proj: Representations with shape ``(T, B, D)``.
            pairs: Disjoint condition-matched indices with shape ``(P, 2)``.
            active: Boolean mask with shape ``(T, P)``.  Active time indices
                receive the mean of separately computed marginal and paired
                difference statistics.
        """

        if pairs is None:
            if active is not None:
                raise ValueError("active requires pairs")
            return SIGReg.forward(self, proj)

        pairs = pairs.to(device=proj.device)
        if active is None:
            active = torch.ones(
                proj.size(0),
                pairs.size(0),
                dtype=torch.bool,
                device=proj.device,
            )
        else:
            active = active.to(device=proj.device)
        self._validate_pairs(proj, pairs, active)
        if pairs.numel() == 0 or not bool(active.any()):
            return SIGReg.forward(self, proj)

        projections = torch.randn(
            proj.size(-1), self.num_proj, device=proj.device
        )
        projections = projections.div_(projections.norm(p=2, dim=0))
        rows = []
        inverse_sqrt_two = 1.0 / math.sqrt(2.0)
        for time_index in range(proj.size(0)):
            marginal = self._projected_statistic(
                proj[time_index].unsqueeze(0),
                projections,
            ).mean()
            selected = active[time_index]
            if not bool(selected.any()):
                rows.append(marginal)
                continue

            selected_pairs = pairs[selected]
            contrasts = (
                proj[time_index, selected_pairs[:, 0]]
                - proj[time_index, selected_pairs[:, 1]]
            ) * inverse_sqrt_two
            if self.randomize_pair_orientation:
                signs = torch.empty(
                    contrasts.size(0),
                    1,
                    device=contrasts.device,
                    dtype=contrasts.dtype,
                )
                signs.bernoulli_(0.5).mul_(2.0).sub_(1.0)
                contrasts = contrasts * signs
            contrast = self._projected_statistic(
                contrasts.unsqueeze(0),
                projections,
            ).mean()
            rows.append(0.5 * (marginal + contrast))
        return torch.stack(rows).mean()


class ScaleCalibratedConditionalSIGReg(ConditionalSIGReg):
    """One SIGReg statistic on replay rows and calibrated pair contrasts.

    Condition-matched observations are usually strongly correlated.  Their
    Haar high-pass contrast therefore has a much smaller natural scale than
    an unconditional embedding.  Applying native SIGReg directly to that
    raw contrast implicitly treats the two observations as independent and
    pushes the local conditional direction toward unit variance.

    This variant divides every active contrast by a frozen reference RMS
    ``contrast_scales[t]`` before concatenating it with rows that do not
    belong to an active pair.  Both kinds of row then have a standard-normal
    null and are scored together by one unchanged SIGReg statistic:

    ``SIGReg(concat(unpaired, (z_i-z_j)/(sqrt(2)*s_t)))``.

    The reference scale must be estimated before optimization from
    model-visible condition pairs and must remain frozen.  No marginal,
    variance, covariance, invariance, or auxiliary prediction loss is
    stacked onto this scalar.
    """

    def __init__(
        self,
        knots=17,
        num_proj=1024,
        randomize_pair_orientation=True,
    ):
        super().__init__(
            knots=knots,
            num_proj=num_proj,
            randomize_pair_orientation=randomize_pair_orientation,
        )

    def forward(
        self,
        proj,
        pairs=None,
        active=None,
        contrast_scales=None,
    ):
        """
        Args:
            proj: Representations with shape ``(T, B, D)``.
            pairs: Disjoint condition-matched indices with shape ``(P, 2)``.
            active: Boolean mask with shape ``(T, P)``.
            contrast_scales: Frozen positive RMS scale for each time index,
                with shape ``(T,)``.  Values at inactive indices are ignored.
        """

        if pairs is None:
            if active is not None or contrast_scales is not None:
                raise ValueError(
                    "active and contrast_scales require pairs"
                )
            return SIGReg.forward(self, proj)

        pairs = pairs.to(device=proj.device)
        if active is None:
            active = torch.ones(
                proj.size(0),
                pairs.size(0),
                dtype=torch.bool,
                device=proj.device,
            )
        else:
            active = active.to(device=proj.device)
        self._validate_pairs(proj, pairs, active)
        if pairs.numel() == 0 or not bool(active.any()):
            return SIGReg.forward(self, proj)
        if contrast_scales is None:
            raise ValueError(
                "contrast_scales are required when any pair is active"
            )
        scales = torch.as_tensor(
            contrast_scales,
            device=proj.device,
            dtype=torch.float32,
        )
        if scales.dim() != 1 or scales.numel() != proj.size(0):
            raise ValueError(
                "contrast_scales must have shape (T,), "
                f"expected {(proj.size(0),)}, got {tuple(scales.shape)}"
            )
        active_times = active.any(dim=1)
        active_scales = scales[active_times]
        if not bool(torch.isfinite(active_scales).all()):
            raise ValueError("active contrast scales must be finite")
        if bool((active_scales <= 0).any()):
            raise ValueError("active contrast scales must be positive")

        projections = torch.randn(
            proj.size(-1), self.num_proj, device=proj.device
        )
        projections = projections.div_(projections.norm(p=2, dim=0))
        rows = []
        inverse_sqrt_two = 1.0 / math.sqrt(2.0)
        for time_index in range(proj.size(0)):
            selected = active[time_index]
            if not bool(selected.any()):
                population = proj[time_index]
            else:
                selected_pairs = pairs[selected]
                contrasts = (
                    proj[time_index, selected_pairs[:, 0]]
                    - proj[time_index, selected_pairs[:, 1]]
                ) * inverse_sqrt_two
                contrasts = contrasts / scales[time_index].to(
                    dtype=contrasts.dtype
                )
                if self.randomize_pair_orientation:
                    signs = torch.empty(
                        contrasts.size(0),
                        1,
                        device=contrasts.device,
                        dtype=contrasts.dtype,
                    )
                    signs.bernoulli_(0.5).mul_(2.0).sub_(1.0)
                    contrasts = contrasts * signs
                unpaired = torch.ones(
                    proj.size(1),
                    dtype=torch.bool,
                    device=proj.device,
                )
                unpaired[selected_pairs.flatten()] = False
                population = torch.cat(
                    [proj[time_index, unpaired], contrasts],
                    dim=0,
                )
            rows.append(
                self._projected_statistic(
                    population.unsqueeze(0),
                    projections,
                ).mean()
            )
        return torch.stack(rows).mean()


class DynamicsResponseSIGReg(SIGReg):
    """One SIGReg statistic on calibrated real and predicted responses.

    The regularizer receives deterministic predictions (callers are
    responsible for disabling inference-time dropout) and real transition
    targets.  At a condition-identifiable transition it replaces matched
    rows by target and prediction Haar contrasts, normalized with a frozen
    source response scale.  At an irreducible transition it includes only
    the real contrast and removes the matched prediction rows.  Unpaired
    replay targets and predictions remain in the same population.

    Every transition population is evaluated with the same random
    projections and the results are averaged into one scalar.  This is not a
    sum of native, conditional, variance, covariance, or prediction losses.

    Args:
        knots: Number of characteristic-function integration knots.
        num_proj: Number of random feature projections.
        reserve_factor: Multiplicative response margin above the frozen
            source contrast RMS.
        randomize_pair_orientation: Apply a shared random Rademacher sign to
            real and predicted contrasts from the same pair.
    """

    def __init__(
        self,
        knots=17,
        num_proj=1024,
        reserve_factor=math.sqrt(2.0),
        randomize_pair_orientation=True,
    ):
        super().__init__(knots=knots, num_proj=num_proj)
        if not math.isfinite(reserve_factor) or reserve_factor <= 0:
            raise ValueError("reserve_factor must be finite and positive")
        self.reserve_factor = float(reserve_factor)
        self.randomize_pair_orientation = randomize_pair_orientation

    @staticmethod
    def _validate_response_contract(
        targets,
        predictions,
        pairs,
        target_active,
        prediction_active,
        contrast_scales,
    ):
        if targets.dim() != 3 or predictions.dim() != 3:
            raise ValueError(
                "targets and predictions must have shape (T, B, D)"
            )
        if targets.shape != predictions.shape:
            raise ValueError(
                "targets and predictions must have identical shapes"
            )
        ConditionalSIGReg._validate_pairs(
            targets,
            pairs,
            target_active,
        )
        expected = target_active.shape
        if tuple(prediction_active.shape) != tuple(expected):
            raise ValueError(
                "prediction_active must have shape (T, P), "
                f"expected {tuple(expected)}, got "
                f"{tuple(prediction_active.shape)}"
            )
        if prediction_active.dtype != torch.bool:
            raise TypeError(
                "prediction_active must use torch.bool, "
                f"got {prediction_active.dtype}"
            )
        if bool((prediction_active & ~target_active).any()):
            raise ValueError(
                "prediction-active pairs must also be target-active"
            )
        if (
            contrast_scales.dim() != 1
            or contrast_scales.numel() != targets.size(0)
        ):
            raise ValueError(
                "contrast_scales must have shape (T,), "
                f"expected {(targets.size(0),)}, got "
                f"{tuple(contrast_scales.shape)}"
            )
        active_times = target_active.any(dim=1)
        active_scales = contrast_scales[active_times]
        if not bool(torch.isfinite(active_scales).all()):
            raise ValueError("active contrast scales must be finite")
        if bool((active_scales <= 0).any()):
            raise ValueError("active contrast scales must be positive")

    def forward(
        self,
        targets,
        predictions,
        pairs=None,
        target_active=None,
        prediction_active=None,
        contrast_scales=None,
    ):
        """
        Args:
            targets: Real transition embeddings with shape ``(T, B, D)``.
            predictions: Deterministic predicted embeddings of the same shape.
            pairs: Disjoint condition-matched batch indices ``(P, 2)``.
            target_active: Real response contrast mask ``(T, P)``.
            prediction_active: Identifiable prediction contrast mask ``(T,P)``.
            contrast_scales: Frozen source contrast RMS with shape ``(T,)``.
        """

        if targets.shape != predictions.shape:
            raise ValueError(
                "targets and predictions must have identical shapes"
            )
        if pairs is None:
            if any(
                value is not None
                for value in (
                    target_active,
                    prediction_active,
                    contrast_scales,
                )
            ):
                raise ValueError(
                    "response masks and scales require pairs"
                )
            return SIGReg.forward(
                self,
                torch.cat([targets, predictions], dim=1),
            )

        pairs = pairs.to(device=targets.device)
        if target_active is None:
            target_active = torch.ones(
                targets.size(0),
                pairs.size(0),
                dtype=torch.bool,
                device=targets.device,
            )
        else:
            target_active = target_active.to(device=targets.device)
        if prediction_active is None:
            prediction_active = target_active.clone()
        else:
            prediction_active = prediction_active.to(
                device=targets.device
            )
        if contrast_scales is None:
            raise ValueError("contrast_scales are required with pairs")
        scales = torch.as_tensor(
            contrast_scales,
            device=targets.device,
            dtype=torch.float32,
        )
        self._validate_response_contract(
            targets,
            predictions,
            pairs,
            target_active,
            prediction_active,
            scales,
        )
        if pairs.numel() == 0 or not bool(target_active.any()):
            return SIGReg.forward(
                self,
                torch.cat([targets, predictions], dim=1),
            )

        projections = torch.randn(
            targets.size(-1),
            self.num_proj,
            device=targets.device,
        )
        projections = projections.div_(projections.norm(p=2, dim=0))
        rows = []
        base_scale = math.sqrt(2.0) * self.reserve_factor
        for time_index in range(targets.size(0)):
            selected_target = target_active[time_index]
            selected_prediction = prediction_active[time_index]
            if not bool(selected_target.any()):
                population = torch.cat(
                    [targets[time_index], predictions[time_index]],
                    dim=0,
                )
            else:
                target_pairs = pairs[selected_target]
                target_unpaired = torch.ones(
                    targets.size(1),
                    dtype=torch.bool,
                    device=targets.device,
                )
                target_unpaired[target_pairs.flatten()] = False
                denominator = (
                    scales[time_index].to(dtype=targets.dtype)
                    * base_scale
                )
                target_contrasts = (
                    targets[time_index, target_pairs[:, 0]]
                    - targets[time_index, target_pairs[:, 1]]
                ) / denominator
                pair_signs = None
                if self.randomize_pair_orientation:
                    pair_signs = torch.empty(
                        pairs.size(0),
                        1,
                        dtype=targets.dtype,
                        device=targets.device,
                    )
                    pair_signs.bernoulli_(0.5).mul_(2.0).sub_(1.0)
                    target_contrasts = (
                        target_contrasts * pair_signs[selected_target]
                    )
                target_population = torch.cat(
                    [
                        targets[time_index, target_unpaired],
                        target_contrasts,
                    ],
                    dim=0,
                )

                prediction_unpaired = torch.ones(
                    predictions.size(1),
                    dtype=torch.bool,
                    device=predictions.device,
                )
                # Prediction rows from any target-active matched pair are
                # removed.  They return only through an identifiable
                # prediction contrast.
                prediction_unpaired[target_pairs.flatten()] = False
                prediction_parts = [
                    predictions[time_index, prediction_unpaired]
                ]
                if bool(selected_prediction.any()):
                    prediction_pairs = pairs[selected_prediction]
                    prediction_contrasts = (
                        predictions[
                            time_index,
                            prediction_pairs[:, 0],
                        ]
                        - predictions[
                            time_index,
                            prediction_pairs[:, 1],
                        ]
                    ) / denominator
                    if pair_signs is not None:
                        prediction_contrasts = (
                            prediction_contrasts
                            * pair_signs[selected_prediction]
                        )
                    prediction_parts.append(prediction_contrasts)
                population = torch.cat(
                    [target_population, *prediction_parts],
                    dim=0,
                )
            rows.append(
                self._projected_statistic(
                    population.unsqueeze(0),
                    projections,
                ).mean()
            )
        return torch.stack(rows).mean()


class VISRegLoss(torch.nn.Module):
    """VISReg sliced Gaussianity regularizer.

    This implementation is vendored from ``stable-pretraining`` commit
    ``7830274c5b92637da7b1a433766494df0d5dbe85``. See
    ``THIRD_PARTY_NOTICES.md`` for source and license details.

    For each leading slice, the batch of embeddings is regularized toward an
    isotropic standard Gaussian through three terms:

    - center: penalize non-zero feature means;
    - scale: penalize per-feature standard deviations away from one;
    - shape: match sorted random one-dimensional projections to theoretical
      standard-normal quantiles.

    Args:
        num_projections: Number of random one-dimensional projections.
        lambda_scale: Weight on the unit-scale term.
        lambda_shape: Weight on the sliced-quantile shape term.
        lambda_center: Weight on the zero-mean term.
    """

    def __init__(
        self,
        num_projections: int = 256,
        lambda_scale: float = 1.0,
        lambda_shape: float = 1.0,
        lambda_center: float = 1.0,
    ):
        super().__init__()
        self.K = num_projections
        self.lambda_scale = lambda_scale
        self.lambda_shape = lambda_shape
        self.lambda_center = lambda_center
        self._cached_B = -1
        self._cached_target = None

    def _get_target(self, B: int, device, dtype) -> torch.Tensor:
        """Return theoretical standard-normal quantiles for ``B`` samples."""

        if self._cached_B != B:
            q = torch.linspace(
                1,
                B,
                B,
                device=device,
                dtype=torch.float32,
            ) / (B + 1)
            self._cached_target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2))
            self._cached_B = B
        return self._cached_target.to(device=device, dtype=dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute VISReg for embeddings shaped ``(V, B, D)``."""

        _, B, D = z.shape

        mu = z.mean(dim=1, keepdim=True)
        center_loss = mu.pow(2).mean()

        z_centered = z - mu
        std = z_centered.norm(dim=1).div(math.sqrt(B)) + 1e-6
        scale_loss = (std - 1.0).pow(2).mean()

        z_norm = z_centered / std.detach().unsqueeze(1)
        W = F.normalize(
            torch.randn(D, self.K, device=z.device, dtype=z.dtype),
            dim=0,
        )
        p_sorted = (z_norm @ W).sort(dim=1).values
        target = self._get_target(B, z.device, z.dtype).view(1, B, 1)
        shape_loss = (p_sorted - target).pow(2).mean()

        return (
            self.lambda_scale * scale_loss
            + self.lambda_shape * shape_loss
            + self.lambda_center * center_loss
        )


class VCReg(torch.nn.Module):
    """Variance-Covariance Regularizer"""

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def _std_loss(self, z):
        z = z.transpose(0, 1)  # (T, B, D)
        std = (z.var(dim=1) + self.eps).sqrt()  # (T, D)
        std_loss = torch.mean(F.relu(1 - std), dim=-1)  # (T,)
        return std_loss

    def _cov_loss(self, z):
        B, T, D = z.shape
        z = z.transpose(0, 1)  # (T, B, D)
        cov = einsum(z, z, 't b i, t b j -> t i j') / (B - 1)  # (T, D, D)
        diag = einsum(cov, 't i i -> t i').pow(2).sum(dim=-1)  # (T,)
        cov_loss = (cov.pow(2).sum(dim=[-1, -2]) - diag).div(D**2 - D)  # (T,)
        return cov_loss

    def forward(self, z):
        """
        z: (..., D)
        """

        if z.dim() == 2:
            D = z.size(-1)
            z = z.view(-1, D)

        z = z - z.mean(
            dim=0, keepdim=True
        )  # mean for each dim across batch samples

        return {
            'std_loss': self._std_loss(z).mean(),
            'std_t_loss': self._std_loss(z.transpose(0, 1)).mean(),
            'cov_loss': self._cov_loss(z).mean(),
            'cov_t_loss': self._cov_loss(z.transpose(0, 1)).mean(),
        }


class PLDMLoss(torch.nn.Module):
    """VCReg anti-collapse + Temporal Alignment + Inverse Dynamics Modeling losses
    reference: https://arxiv.org/abs/2502.14819
    """

    def __init__(self):
        super().__init__()
        self.vc_reg = VCReg()

    def forward(self, z, a_pred=None, a_target=None):
        """
        z: (B, T, D)
        a_pred: (B, T-1, A)
        a_target: (B, T-1, A)
        """

        output = {}
        if a_pred is not None and a_target is not None:
            output['idm_loss'] = F.mse_loss(a_pred, a_target)

        output['temp_align_loss'] = F.mse_loss(z[:, :-1], z[:, 1:])  # detach?
        output.update(self.vc_reg(z))

        return output


class TemporalStraighteningLoss(torch.nn.Module):
    """Temporal Straightening Loss Module (Mean Pairwise Negative Cosine Similarity)
    reference: https://arxiv.org/abs/2603.12231
    """

    def __init__(self):
        super().__init__()
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        v = x[:, 1:] - x[:, :-1]  # velocities
        sim = self.cos_sim(v[:, :-1], v[:, 1:])
        return -sim.mean()


__all__ = [
    'ConditionalSIGReg',
    'DynamicsResponseSIGReg',
    'GroupBalancedSIGReg',
    'JointTemporalCovarianceSIGReg',
    'ScaleCalibratedConditionalSIGReg',
    'PLDMLoss',
    'SIGReg',
    'MixedRadiusTaggedAnchoredJointCFSIGReg',
    'SemanticBlockTaggedAnchoredJointSWReg',
    'TaggedAnchoredJointCFSIGReg',
    'TemporallyCenteredSIGReg',
    'TemporalStraighteningLoss',
    'VCReg',
    'VISRegLoss',
]
