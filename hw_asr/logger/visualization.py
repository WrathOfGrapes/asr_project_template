from .tensorboard import TensorboardWriter
from .wandb import WanDBWriter


def get_visualizer(config, logger, type):
    if type == "tensorboard":
        return TensorboardWriter(config.log_dir, logger, True)

    if type == 'wandb':
        return WanDBWriter(config, logger)

    return None

