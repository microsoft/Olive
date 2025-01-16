from pathlib import Path
from logging import getLogger
import numpy as np
from torch import from_numpy
from torch.utils.data import DataLoader, Dataset
from olive.data.registry import Registry

logger = getLogger(__name__)

class ImagenetDataset(Dataset):
  def __init__(self, data_dir: str):
    with np.load(Path(data_dir) / '_data.npz') as data:
      self.images = from_numpy(data['images'])
      self.labels = from_numpy(data['labels'])

  def __len__(self):
    return min(len(self.images), len(self.labels))

  def __getitem__(self, idx):
    return {"input": self.images[idx]}, self.labels[idx]


@Registry.register_dataloader()
def imagenet_dataloader(dataset, batch_size, data_dir, **kwargs):
  return DataLoader(ImagenetDataset(data_dir), batch_size=batch_size, shuffle=False)

@Registry.register_post_process()
def imagenet_post_fun(output):
  return output.argmax(axis=1)

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dataset_model_download import download_dataset, download_classfile, download_resnet_model
from datasets import Split
from pathlib import Path

model_dir = Path(f'./models').resolve()
model_dir.mkdir(parents=True, exist_ok=True)

model_name = 'resnet50'
model_path = model_dir / f'{model_name}.pth'

if model_path.exists():
  logger.debug('model already downloaded.')
else:
  download_resnet_model(model_path, model_name)

data_dir = Path('./data/imagenet').resolve()
data_dir.mkdir(parents=True, exist_ok=True)

data_ready = data_dir / '.complete'
if data_ready.exists():
  logger.debug('Dataset already downloaded.')
else:
  download_dataset(data_dir, Split.VALIDATION, size=256)
  download_classfile(data_dir)
  data_ready.touch()