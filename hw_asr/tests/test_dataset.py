import unittest

from hw_asr.datasets import LibrispeechDataset, CustomDirAudioDataset
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser


class TestDataset(unittest.TestCase):
    def test_librispeech(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        config_parser = ConfigParser.get_default_configs()

        ds = LibrispeechDataset(
            "dev-clean",
            text_encoder=text_encoder,
            config_parser=config_parser,
            max_text_length=140,
            max_audio_length=13,
            limit=10,
        )
        item = ds[0]
        print(item)

    def test_custom_dataset(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        config_parser = ConfigParser.get_default_configs()

        audio_dir = str(ROOT_PATH / "test_data" / "audio")
        transc_dir = str(ROOT_PATH / "test_data" / "transcriptions")

        ds = CustomDirAudioDataset(
            audio_dir,
            transc_dir,
            text_encoder=text_encoder,
            config_parser=config_parser,
            limit=1,
            max_audio_length=8,
            max_text_length=130,
        )
        item = ds[0]
        print(item)
