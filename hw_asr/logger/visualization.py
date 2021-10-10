from .tensorboard import TensorboardWriter
from .wandb import WanDBdWriter


def get_visualizer(config, logger, type):
    if type == "tensorboard":
        return TensorboardWriter(config.log_dir, logger, True)

    if type == 'wandb':
        return WanDBdWriter(config, logger)

    return None

