import shutil
import unittest
from pathlib import Path

import PIL
import numpy as np
import torch
import torchaudio
from torchvision.transforms import ToTensor

from hw_asr.logger.tensorboard import TensorboardWriter
from hw_asr.logger.utils import plot_spectrogram_to_buf
from hw_asr.logger.wandb import WanDBWriter
from hw_asr.utils.parse_config import ConfigParser


class TestVisualization(unittest.TestCase):
    def test_visualizers(self):
        log_dir = str(Path(__file__).parent / "logs_dir")

        try:
            config = ConfigParser.get_default_configs()
            logger = config.get_logger("test")

            tensorboard = TensorboardWriter(log_dir, logger, True)
            wandb = WanDBWriter(config, logger)

            audio_path = Path(__file__).parent.parent.parent / "test_data" / "audio" / "84-121550-0000.flac"
            audio, sr = torchaudio.load(audio_path)

            wave2spec = config.init_obj(
                config["preprocessing"]["spectrogram"],
                torchaudio.transforms,
            )

            wave = wave2spec(audio)
            image = ToTensor()(PIL.Image.open(plot_spectrogram_to_buf(wave.squeeze(0).log())))

            hist = torch.from_numpy(np.asarray([1, 2, 3, 4]))

            test_data = [
                1,
                {"test1": 1, "test2": 2},
                image,
                audio,
                "test",
                hist
            ]

            test_methods = [
                "add_scalar",
                "add_scalars",
                "add_image",
                "add_audio",
                "add_text",
                "add_histogram"
            ]

            for method, value in zip(test_methods, test_data):
                kwargs = {}
                if method == 'add_audio':
                    kwargs = {'sample_rate': sr}
                elif method == 'add_histogram':
                    kwargs = {'bins': 'auto'}

                logger.info(f"test {method}")
                getattr(tensorboard, method)(method, value, **kwargs)
                getattr(wandb, method)(method, value, **kwargs)

        finally:
            shutil.rmtree(log_dir)



