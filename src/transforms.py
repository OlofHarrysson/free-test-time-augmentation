from torchvision import transforms
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image


def get_train_transforms():
  transformer = Transformer()

  return transforms.Compose([
      transformer,
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
  ])


def get_val_transforms():
  transformer = Transformer()

  return transforms.Compose([
      transformer,
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
  ])


class Transformer():
  def __init__(self):
    self.seq = iaa.Resize({"height": 64, "width": 64})

  def __call__(self, im):
    return im
