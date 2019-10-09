from torchvision import datasets
from torch.utils.data import DataLoader
from ..transforms import get_train_transforms, get_val_transforms


def get_trainloader(config):
  transforms = get_train_transforms()
  dataset = DaDataset('src/data/datasets', transforms, split='train')
  return DataLoader(dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.num_workers)


def get_valloader(config):
  transforms = get_val_transforms()
  dataset = DaDataset('src/data/datasets', transforms, split='test')
  return DataLoader(dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers)


class DaDataset(datasets.STL10):
  def __init__(self, path, transforms, split='train'):
    super().__init__(path, split, download=False)
    self.transforms = transforms

  def __getitem__(self, index):
    im, label = super().__getitem__(index)
    return self.transforms(im), label


# class DaDataset(datasets.CIFAR10):
#   def __init__(self, path, transforms, split=True):
#     super().__init__(path, split, download=False)
#     self.transforms = transforms

#   def __getitem__(self, index):
#     im, label = super().__getitem__(index)
#     return self.transforms(im), label
