import torch
import torchaudio
import librosa
from torchaudio import transforms as T
from torch.utils.data import Dataset
import pandas as pd
import math
import os
import args
from utils import Normalize

METADATA_CSV = 'metadata.csv'
DESIRED_DURATION = 8  # only 15 respiratory cycles have a length >= 8 secs, and the 5 cycles that have a length >= 9 secs contain artefacts towards the end
DESIRED_SR = 16000  # sampling rate
SPRS_CLASS_DICT = {'Normal': 0, 'Fine Crackle': 1, 'Wheeze': 2, 'Coarse Crackle': 3, 'Wheeze+Crackle': 4, 'Rhonchi': 5,
                   'Stridor': 6}

DEFAULT_NFFT = 1024
DEFAULT_NMELS = 64
DEFAULT_WIN_LENGTH = 1024
DEFAULT_HOP_LENGTH = 512
DEFAULT_FMIN = 50
DEFAULT_FMAX = 2000
# ICBHI label mapping
"""
LABEL_N, LABEL_C, LABEL_W, LABEL_B = 0, 1, 2, 3
label 0 for normal respiration
label 1 for crackles
label 2 for wheezes
label 3 for both
"""


class ICBHI(Dataset):
    def __init__(self, data_path, split, metadatafile=METADATA_CSV, duration=DESIRED_DURATION, samplerate=DESIRED_SR,
                 device="cpu", fade_samples_ratio=16, pad_type="circular"):

        self.data_path = data_path
        self.csv_path = os.path.join(self.data_path, metadatafile)
        self.split = split
        self.df = pd.read_csv(self.csv_path)
        if self.split == 'train':
            self.df = self.df[(self.df["split"] == self.split)]
        elif self.split == 'test':
            self.df = self.df[(self.df["split"] == self.split)]
        self.duration = duration
        self.samplerate = samplerate
        self.targetsample = self.duration * self.samplerate
        self.pad_type = pad_type
        self.device = device
        self.fade_samples_ratio = fade_samples_ratio
        self.fade_samples = int(self.samplerate / self.fade_samples_ratio)
        self.fade = T.Fade(fade_in_len=self.fade_samples, fade_out_len=self.fade_samples, fade_shape='linear')
        self.fade_out = T.Fade(fade_in_len=0, fade_out_len=self.fade_samples, fade_shape='linear')
        self.pth_path = os.path.join(self.data_path, "icbhi-4" + str(self.split) + '_duration' + str(
                self.duration) + ".pth")

        if os.path.exists(self.pth_path):
            print(f"Loading dataset {self.split}...")
            pth_dataset = torch.load(self.pth_path)
            # self.data, self.labels, self.metadata_labels = pth_dataset['data'].to(self.device), pth_dataset['label'].to(self.device), pth_dataset['meta_label'].to(self.device)
            self.data, self.labels= pth_dataset['data'], pth_dataset['label']
            print(f"Dataset {self.split} loaded !")
        else:
            print(f"File {self.pth_path} does not exist. Creating dataset...")
            self.data, self.labels= self.get_dataset()
            data_dict = {"data": self.data, "label": self.labels}
            # self.data, self.labels, self.metadata_labels = self.data.to(self.device), self.labels.to(self.device), self.metadata_labels.to(self.device)
            print(f"Dataset {self.split} created !")
            torch.save(data_dict, self.pth_path)
            print(f"File {self.pth_path} Saved!")

    def get_sample(self, i):

        ith_row = self.df.iloc[i]
        filepath = ith_row['filepath']
        filepath = os.path.join(self.data_path, filepath)
        onset = ith_row['onset']
        offset = ith_row['offset']
        bool_wheezes = ith_row['wheezes']
        bool_crackles = ith_row['crackles']
        # chest_loc = filepath[4:7]
        # rec_equip = ith_row['device']
        # metalabel_colname = str(self.meta_label)
        # metadata_label = ith_row[metalabel_colname]
        # metadata_label = ith_row['sc_class_num']

        if not bool_wheezes:
            if not bool_crackles:
                label = 0
            else:
                label = 1
        else:
            if not bool_crackles:
                label = 2
            else:
                label = 3

        sr = librosa.get_samplerate(filepath)
        audio, _ = torchaudio.load(filepath, int(onset * sr), (int(offset * sr) - int(onset * sr)))
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != self.samplerate:
            resample = T.Resample(sr, self.samplerate)
            audio = resample(audio)

        return self.fade(audio), label

    def get_dataset(self):

        dataset = []  # 空列表,用于存储处理后的音频样本
        labels = []
        # metadata_labels = []
        # rec_equips = []

        for i in range(len(self.df)):
            audio, label = self.get_sample(i)
            if audio.shape[-1] > self.targetsample:
                audio = audio[..., :self.targetsample]
            else:
                if self.pad_type == 'circular':
                    ratio = math.ceil(self.targetsample / audio.shape[-1])
                    audio = audio.repeat(1, ratio)
                    audio = audio[..., :self.targetsample]
                    audio = self.fade_out(audio)
                elif self.pad_type == 'zero':
                    tmp = torch.zeros(1, self.targetsample, dtype=torch.float32)
                    diff = self.targetsample - audio.shape[-1]
                    tmp[..., diff // 2:audio.shape[-1] + diff // 2] = audio
                    audio = tmp
            dataset.append(audio)
            labels.append(label)
            # metadata_labels.append(metadata_label)
            # rec_equips.append(rec_equip)

        return torch.unsqueeze(torch.vstack(dataset), 1), torch.tensor(labels)  # rec_equips

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        melspec = T.MelSpectrogram(n_fft=DEFAULT_NFFT, n_mels=DEFAULT_NMELS, win_length=DEFAULT_WIN_LENGTH,
                                   hop_length=DEFAULT_HOP_LENGTH, f_min=DEFAULT_FMIN, f_max=DEFAULT_FMAX)
        self.mel_output = melspec(self.data)
        # print("mel_output shape in __getitem__:", self.mel_output[idx].shape)
        return self.data[idx], self.labels[idx], self.mel_output[idx]