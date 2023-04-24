import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torchaudio
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.utils.data_utils import load_yaml


class Collate:
    def __init__(self) -> None:
        super(Collate, self).__init__()

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, ...]:
        """
        :param batch: is a list of tuples of [features, label], where features has dimensions [n_features, length]
        "returns features, lengths, labels:
              features is a Tensor [batchsize, features, max_length]
              labels is a Tesnor of targets [batchsize]
              lengths is a Tensor of lengths [batchsize]
        """

        features = []
        labels = torch.LongTensor(len(batch)).zero_()
        lengths = torch.LongTensor(len(batch)).zero_()
        for i, (feature, label, length) in enumerate(batch):
            features.append(torch.Tensor(feature))
            labels[i] = label
            lengths[i] = length
        features = pad_sequence(features, batch_first=True)

        return features, labels, lengths


class AudioMNISTDataModule(LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.dataset = load_dataset(self.cfg["Base"]["training_dataset"])
        self._transform = self.get_spectrogram_transform()

    def setup(self, stage=None):
        self.collate = Collate()
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset["train"].map(
                self.prepare_features,
                remove_columns=self.dataset.column_names["train"],
                num_proc=6,
                cache_file_name=os.path.join("data/cache", "train_cache.arrow"),
            )
            self.val_dataset = self.dataset["valid"].map(
                self.prepare_features, remove_columns=self.dataset.column_names["valid"]
            )

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset["test"].map(
                self.prepare_features, remove_columns=self.dataset.column_names["test"]
            )

    def train_dataloader(self):
        train_dataset = AudioMNISTDataset(self.train_dataset)
        return DataLoader(
            train_dataset,
            batch_size=self.cfg["Data"]["batch_size"],
            num_workers=self.cfg["Data"]["num_workers"],
            drop_last=True,
            shuffle=True,
            collate_fn=self.collate,
        )

    def val_dataloader(self):
        val_dataset = AudioMNISTDataset(self.val_dataset)
        return DataLoader(
            val_dataset,
            batch_size=self.cfg["Data"]["batch_size"],
            num_workers=self.cfg["Data"]["num_workers"],
            collate_fn=self.collate,
        )

    def test_dataloader(self):
        test_dataset = AudioMNISTDataset(self.test_dataset)
        return DataLoader(
            test_dataset,
            batch_size=self.cfg["Data"]["batch_size"],
            num_workers=self.cfg["Data"]["num_workers"],
            collate_fn=self.collate,
        )

    def get_spectrogram_transform(self):
        # (freq_bins, time_frames),
        audio_transforms = nn.Sequential(
            torchaudio.transforms.Spectrogram(
                n_fft=self.cfg["Data"]["n_fft"],
                hop_length=self.cfg["Data"]["hop_length"],
                power=self.cfg["Data"]["power"],
            ),
            torchaudio.transforms.AmplitudeToDB(),
        )
        return audio_transforms

    def prepare_features(self, examples):
        audio_path = examples["audio"]["path"]
        x, sr = torchaudio.load(audio_path)

        effects = [
            ["gain", "-n", "0"],  # apply 10 db attenuation
            ["remix", "-"],  # merge all the channels
            ["rate", str(self.cfg["Data"]["target_sr"])],
        ]
        effects_x, _ = torchaudio.sox_effects.apply_effects_tensor(x, sr, effects)
        effects_x = effects_x[0]
        examples["feature"] = self._transform(effects_x).T
        examples["label"] = examples["digit"]
        examples["audio_len"] = effects_x.shape[-1]
        return examples


class AudioMNISTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        # x: audio features
        # y: target digit class
        sample = self.data[idx]
        x = sample["feature"]
        y = sample["label"]
        audio_len = sample["audio_len"]
        return x, y, audio_len

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    config = load_yaml("src/config/training.yaml")
    test_audio_module = AudioMNISTDataModule(config)
    test_audio_module.setup("fit")
    train_loader = test_audio_module.train_dataloader()
    xs, ls, ys = next(iter(train_loader))
    print(xs.size(), ls.size(), ys.size())
    print(xs.dtype, ls.dtype, ys.dtype)
