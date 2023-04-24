import torch
import torchaudio
from typing import List, Tuple

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import LightningDataModule


class AudioMNISTDataModule(LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.batch_size = config["Data"]["batch_size"]
        self.num_workers = config["Data"]["num_workers"]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.AudioMNIST('.', train=True, transform=None)
            self.val_dataset = datasets.AudioMNIST('.', train=False, transform=None)

        if stage == 'test' or stage is None:
            self.test_dataset = datasets.AudioMNIST('.', train=False, transform=None)

    def train_dataloader(self):
        train_dataset = AudioMNISTDataset(self.train_dataset.data, self.train_dataset.targets,
                                          transform=transforms.Compose([
                                              torchaudio.transforms.MelSpectrogram(sample_rate=8000, n_mels=128)
                                          ]))
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        val_dataset = AudioMNISTDataset(self.val_dataset.data, self.val_dataset.targets,
                                        transform=transforms.Compose([
                                            torchaudio.transforms.MelSpectrogram(sample_rate=8000, n_mels=128)
                                        ]))
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        test_dataset = AudioMNISTDataset(self.test_dataset.data, self.test_dataset.targets,
                                         transform=transforms.Compose([
                                             torchaudio.transforms.MelSpectrogram(sample_rate=8000, n_mels=128)
                                         ]))
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_spectrogram_transform(self):
        # (freq_bins, time_frames),
        train_audio_transforms = nn.Sequential(
            torchaudio.transforms.Spectrogram(
                n_fft=self.config["Data"]["n_fft"],
                win_length=self.config["Data"]["win_length"],
                hop_length=self.config["Data"]["hop_length"],
                power=self.config["Data"]["power"],
            ),
            torchaudio.transforms.AmplitudeToDB()
        )
        test_audio_transforms = nn.Sequential(
            torchaudio.transforms.Spectrogram(
                n_fft=self.config["Data"]["n_fft"],
                win_length=self.config["Data"]["win_length"],
                hop_length=self.config["Data"]["hop_length"],
                power=self.config["Data"]["power"],
            ),
            torchaudio.transforms.AmplitudeToDB()
        )
        return train_audio_transforms, test_audio_transforms

class AudioMNISTDataset:
    ...