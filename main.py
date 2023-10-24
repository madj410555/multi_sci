import os
import torch
import torch.nn as nn
from torchaudio import transforms as T
import torch.nn.functional as F
from torchinfo import summary
from augmentations import SpecAugment
from models import CNN6, LinearClassifier
from dataset import ICBHI
from utils import Normalize, Standardize
from losses import SupConLoss, SupConCELoss
from ce import train_ce
from hybrid import train_supconce
from args import args
from multihead import MultiModal

if not (os.path.isfile(os.path.join(args.datapath, args.metadata))):
    raise (IOError(f"CSV file {args.metadata} does not exist in {args.datapath}"))

METHOD = args.method

DEFAULT_NUM_CLASSES = 4
DEFAULT_OUT_DIM = 128  # for ssl embedding space dimension
DEFAULT_NFFT = 1024
DEFAULT_NMELS = 64
DEFAULT_WIN_LENGTH = 1024
DEFAULT_HOP_LENGTH = 512
DEFAULT_FMIN = 50
DEFAULT_FMAX = 2000

# Model definition
model= MultiModal().to(args.device)
s = summary(model, device=args.device)
nparams = s.trainable_params

#  Spectrogram definition
# melspec = T.MelSpectrogram(n_fft=DEFAULT_NFFT, n_mels=DEFAULT_NMELS, win_length=DEFAULT_WIN_LENGTH,
#                            hop_length=DEFAULT_HOP_LENGTH, f_min=DEFAULT_FMIN, f_max=DEFAULT_FMAX).to(args.device)
# normalize = Normalize()
# melspec = torch.nn.Sequential(melspec, normalize)
standardize = Standardize(device=args.device)

# Data transformations
specaug = SpecAugment(freq_mask=args.freqmask, time_mask=args.timemask, freq_stripes=args.freqstripes,
                      time_stripes=args.timestripes).to(args.device)
train_transform = nn.Sequential(specaug, standardize)
val_transform = nn.Sequential(standardize)

#  Dataset and dataloaders
train_ds = ICBHI(data_path=args.datapath, metadatafile=args.metadata, duration=args.duration, split='train',
                     device=args.device, samplerate=args.samplerate, pad_type=args.pad)
val_ds = ICBHI(data_path=args.datapath, metadatafile=args.metadata, duration=args.duration, split='test',
                   device=args.device, samplerate=args.samplerate, pad_type=args.pad)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.bs, shuffle=False)


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)

if args.dataset == 'ICBHI':
    if args.noweights:
        criterion_ce = nn.CrossEntropyLoss()
    else:
        weights = torch.tensor([2063, 1215, 501, 363],
                               dtype=torch.float32)  # N_COUNT, C_COUNT, W_COUNT, B_COUNT = 2063, 1215, 501, 363 for trainset
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        weights = weights.to(args.device)
        criterion_ce = nn.CrossEntropyLoss(weight=weights)
else:
    criterion_ce = nn.CrossEntropyLoss()


history = train_ce(model, train_loader, val_loader, train_transform, val_transform, criterion_ce, optimizer,
                       args.epochs, scheduler)
del model

del train_ds;
del val_ds
del train_loader;
del val_loader