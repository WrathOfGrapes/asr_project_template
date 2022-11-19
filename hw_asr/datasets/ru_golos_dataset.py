import json
import logging
import os
import shutil
from pathlib import Path

import jsonlines
import pandas as pd
import torchaudio
from hw_asr.base.base_dataset import BaseDataset
from hw_asr.utils import ROOT_PATH
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

logger = logging.getLogger(__name__)

URL_LINKS = {
    "farfield": "https://sc.link/1Z3",
    "train_1": "https://sc.link/MvQ",
    "train_2": "https://sc.link/NwL",
    "train_3": "https://sc.link/Oxg",
    "train_4": "https://sc.link/Pyz",
    "train_5": "https://sc.link/Qz7",
    "train_6": "https://sc.link/RAL",
    "train_7": "https://sc.link/VG5",
    "train_8": "https://sc.link/WJW",
    "train_9": "https://sc.link/XKk", 
}

class GolosDataset(BaseDataset):
    def __init__(self, part, names=["crowd7", "crowd8", "crowd9"], data_dir=None, *args, **kwargs):
        """
        :param part: which part of dataset to use (only train is supported)
        :param names: which part of train split to use (crowd{i} or farfield),
            crowd0 is not supported
        :param data_dir: Path object with the path to data folder
        """
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ru_golos"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(part, names)

        super().__init__(index, *args, **kwargs)

    def _load_dataset(self, name):
        print(f"Loading GOLOS_{name}")

        if (self._data_dir / "train" / name).exists():
            return
        if (self._data_dir / "train" / "crowd" / f"{name[-1]}").exists():
            return

        if name == "farfield":
            url_name = name
        else:
            url_name = f"train_{name[-1]}"

        arch_path = self._data_dir / f"{url_name}.tar"
        if not arch_path.exists():
            download_file(URL_LINKS[url_name], arch_path)
            shutil.unpack_archive(arch_path, self._data_dir)
        if name[-1] == "9":
            shutil.move(str(self._data_dir / "train" / "manifest.jsonl"),\
                        str(self._data_dir / "manifest.jsonl"))
        os.remove(str(arch_path))

    def _get_or_load_index(self, part, names):
        index_path = self._data_dir / f"{part}_{'_'.join(names)}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part, names)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part, names):
        index = []
        split_dir = self._data_dir / part
        for name in names:
            if name == "farfield":
                if not (split_dir / name).exists():
                    self._load_dataset(name)
            elif not (split_dir / "crowd" / f"{name[-1]}").exists():
                self._load_dataset(name)

        wav_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".wav") for f in filenames]):
                wav_dirs.add(dirpath)
        for wav_dir in tqdm(
                list(wav_dirs), desc=f"Preparing golos folders: {part}"
        ):
            wav_dir = Path(wav_dir)
            trans_path = self._data_dir / "manifest.jsonl"
            assert trans_path.exists(), "download crowd9 first"
            with jsonlines.open(str(trans_path)) as reader:
                for obj in reader.iter(type=dict):
                    if "farfield" not in str(wav_dir):
                        path_check = f"crowd/{str(wav_dir)[-1]}"
                        if f"crowd{str(wav_dir)[-1]}" not in names:
                            continue
                    else:
                        path_check = "farfield"
                        if "farfield" not in names:
                            continue
                    if  path_check not in obj["audio_filepath"]:
                        continue
                    w_id = obj['id'] + ".wav"
                    w_text = obj['text'].strip()
                    wav_path = wav_dir / w_id
                    t_info = torchaudio.info(str(wav_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(wav_path.absolute().resolve()),
                            "text": w_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index
