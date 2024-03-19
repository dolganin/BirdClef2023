#imports
import os
import timm
import torch
import numpy as np 
import pandas as pd
import librosa as lb
import torch.nn as nn
import soundfile as sf
import yaml

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
import torchvision.transforms as transforms
from  soundfile import SoundFile 
from sklearn.model_selection import train_test_split

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

bs = 32
classnum = 264
n_mels = 224
epochs = 10
lr = 3e-3
