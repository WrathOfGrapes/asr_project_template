import logging
from pathlib import Path

from hw_asr.datasets.custom_audio_dataset import CustomAudioDataset
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(CustomAudioDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + '.txt')
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip()
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)


if __name__ == "__main__":
    text_encoder = CTCCharTextEncoder.get_simple_alphabet()
    config_parser = ConfigParser.get_default_configs()

    ds = CustomDirAudioDataset("data/datasets/custom/audio",
                               text_encoder=text_encoder, config_parser=config_parser)
    item = ds[0]
    print(item)
