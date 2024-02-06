
import os 
import torch
import torchvision

from pathlib import Path
from torchvision.datasets import INaturalist
from torch.utils.data import DataLoader 

def create_dataloaders(train_transforms:torchvision.transforms,
                       test_transforms:torchvision.transforms,
                       batch_size:int=128,
                       num_workers:int=os.cpu_count(),
                       device:torch.device="cpu"):
  """
  Creates train and test dataloaders for INaturalist dataset 
  from torchvision.datasets.

  Keyword Arguments:
    :arg train_transforms: transformation for train data
    :type train_transforms: torchvision.transforms
    :arg test_transforms: transformation for test data
    :type test_transforms: torchvision.transforms
    :arg batch_size: batch size of the data. Default 128.
    :type batch_size: int
    :arg num_workers: number of CPUs to use. Default os.cpu_count().
    :type num_workers: int
    :arg device: device "CPU" or "GPU". Default "CPU"
    :type device: str
  
  Example Usage:
    train_dataloader, test_dataloader, train_data = create_dataloaders(train_transforms=train_transforms,
                                                          test_transforsm=test_transforms,
                                                          batch_size=batch_size,
                                                          num_workers=os.cpu_count(),
                                                          device="cpu")
  """
  current_working_dir = Path(os.getcwd())
  data_dir = current_working_dir / "data"
  train_dir = data_dir / "train"
  test_dir = data_dir / "test"
  
  data_dir.mkdir(parents=True, exist_ok=True)
  train_dir.mkdir(parents=True, exist_ok=True)
  test_dir.mkdir(parents=True, exist_ok=True)

  train_data = INaturalist(root=train_dir,
                          version="2021_train_mini",
                          download=True,
                          transform=train_transforms)
  test_data = INaturalist(root=test_dir,
                          version="2021_valid",
                          download=True,
                          transform=test_transforms)
  
  train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True,
                                generator=torch.Generator(device=device))
  
  test_dataloader = DataLoader(dataset=test_data,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                              generator=torch.Generator(device=device))
  



  return train_dataloader, test_dataloader, train_data
