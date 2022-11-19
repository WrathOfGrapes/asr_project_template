import concurrent.futures as cf
import json
import logging
import os
import shutil
from asyncio import as_completed
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from hw_asr.base.base_dataset import BaseDataset
from hw_asr.utils import ROOT_PATH
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RuCommonVoiceDataset(BaseDataset):
    def __init__(self, part, data_dir=None, use_vad=False, *args, **kwargs):
        """
        :param part: which part of dataset to use
        :param data_dir: Path objecth with the path to data folder
        :param use_vad: whether to preprocess all audios with Voice Activity Detector
            in order to cut silence at the beggining and end of the audio
        """
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ru_commonvoice"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(part, use_vad)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        df = pd.read_csv(str(self._data_dir / f'{part}.tsv'), sep='\t')
        for _, row in df.iterrows():
            f_name = row['path']
            file_path = self._data_dir / 'clips' / f_name
            shutil.move(str(file_path), str(self._data_dir / part / f_name))

    def _load_dataset(self):
        arch_path = self._data_dir / "cv-corpus-11.0-2022-09-21-ru.tar.gz"

        # url wget is not supported due to email confirmation needed
        assert arch_path.exists(), "please download RU Common Voice 11.0 from the official website"
        print(f"Loading RU Common Voice 11.0")

        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "cv-corpus-11.0-2022-09-21/ru").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "cv-corpus-11.0-2022-09-21"))

        (self._data_dir / "train").mkdir(exist_ok=True, parents=True)
        (self._data_dir / "dev").mkdir(exist_ok=True, parents=True)
        (self._data_dir / "test").mkdir(exist_ok=True, parents=True)
        
        self._load_part("train")
        self._load_part("dev")
        self._load_part("test")

        shutil.rmtree(str(self._data_dir / "clips"))


    def _get_or_load_index(self, part, use_vad):
        if use_vad:
            index_path = self._data_dir / f"{part}_vad_index.json"
        else:
            index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part, use_vad)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part, use_vad):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_dataset()

        mp3_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".mp3") for f in filenames]):
                mp3_dirs.add(dirpath)
        for mp3_dir in tqdm(
                list(mp3_dirs), desc=f"Preparing ru common voice folders: {part}"
        ):
            torchaudio.set_audio_backend('sox_io')
            mp3_dir = Path(mp3_dir)
            trans_path = self._data_dir / f"{part}.tsv"
            df = pd.read_csv(trans_path, sep='\t')
            with cf.ThreadPoolExecutor(max_workers=100) as executor: 
                future_to_dict =  {executor.submit(add_to_index, mp3_dir, row, use_vad): row\
                                   for _, row in df.iterrows()}
                for future in cf.as_completed(future_to_dict):
                    index.append(future.result())
        return index    


def add_to_index(mp3_dir, row, use_vad):
    m_id = row['path']
    m_text = row['sentence'].strip()
    mp3_path = mp3_dir / m_id
    if use_vad:
        audio_tensor, sr = torchaudio.load(str(mp3_path))
        # Common voice has too much noise and silence and the start and end
        audio_tensor = torchaudio.functional.vad(audio_tensor, sr, pre_trigger_time=0.15) # cut leading silence
        audio_tensor = torch.flip(audio_tensor, [0, 1])
        audio_tensor = torchaudio.functional.vad(audio_tensor, sr, pre_trigger_time=0.15) # cut ending silence
        audio_tensor = torch.flip(audio_tensor, [0, 1])
        mp3_path = Path(str(mp3_path)[:-4] + "_vad.mp3")
        torchaudio.save(str(mp3_path), audio_tensor, sr)

    t_info = torchaudio.info(str(mp3_path))
    length = t_info.num_frames / t_info.sample_rate
    res_dict= {
              "path": str(mp3_path.absolute().resolve()),
              "text": m_text.lower(),
              "audio_len": length,
    }
    return res_dict
