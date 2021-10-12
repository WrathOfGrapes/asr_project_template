import logging
import random

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            text_encoder: BaseTextEncoder,
            config_parser: ConfigParser,
            wave_augs=None,
            spec_augs=None,
            limit=None,
            max_audio_length=None,
            max_text_length=None,
    ):
        self.text_encoder = text_encoder
        self.config_parser = config_parser
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs

        for entry in index:
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - duration of audio (in seconds)."
            )
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
            assert "text" in entry, (
                "Each dataset item should include field 'text'"
                " - text transcription of the audio."
            )

        index = self._filter_records_from_dataset(
            index, max_audio_length, max_text_length, limit
        )

        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave = self.load_audio(audio_path)
        audio_wave, audio_spec = self.process_wave(audio_wave)
        return {
            "audio": audio_wave,
            "spectrogram": audio_spec,
            "duration": data_dict["audio_len"],
            "text": data_dict["text"],
            "text_encoded": self.text_encoder.encode(data_dict["text"]),
            "audio_path": audio_path,
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)
            wave2spec = self.config_parser.init_obj(
                self.config_parser["preprocessing"]["spectrogram"],
                torchaudio.transforms,
            )
            audio_tensor_spec = wave2spec(audio_tensor_wave)
            if self.spec_augs is not None:
                audio_tensor_spec = self.spec_augs(audio_tensor_spec)
            return audio_tensor_wave, audio_tensor_spec

    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length, max_text_length, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = set(
                [el['path'] for el in index if el["audio_len"] >= max_audio_length]
            )
            _total = len(exceeds_audio_length)
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        if max_text_length is not None:
            exceeds_text_length = set(
                [
                    el['path']
                    for el in index
                    if len(BaseTextEncoder.normalize_text(el["text"])) >= max_text_length
                ]
            )
            _total = len(exceeds_text_length)
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_text_length} characters. Excluding them."
            )
        else:
            exceeds_text_length = False

        records_to_filter = exceeds_text_length | exceeds_audio_length

        if records_to_filter is not False:
            _total = len(records_to_filter)
            index = [el for el in index if el['path'] not in records_to_filter]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records from dataset"
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index
