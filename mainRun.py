import os
import cv2
import torch
import imutils
from PIL import Image
from model import Generator
from model import Discriminator
from torchvision import transforms as T
from torchvision.utils import save_image


# ---- General Settings -----------------
use_gpu = torch.cuda.is_available()  # use GPU
if use_gpu:
    device = torch.device("cuda:0")  # or torch.device('cuda')
else:
    device = torch.device("cpu")



