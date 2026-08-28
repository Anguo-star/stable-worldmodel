"""VIS-WM training entry.

VIS-WM intentionally reuses LeWM's model implementation and training loop.
Its independent entry and config keep the VISReg objective, run identity, and
public command surface distinct from native LeWM.
"""

import hydra
from stable_worldmodel.wm.loss import VISRegLoss

if __package__:
    from .lewm import run_training
else:
    from lewm import run_training


VISWM_REGULARIZERS = {'visreg': VISRegLoss}


@hydra.main(version_base=None, config_path='./config', config_name='viswm')
def run(cfg):
    return run_training(
        cfg,
        regularizers=VISWM_REGULARIZERS,
    )


if __name__ == '__main__':
    run()
