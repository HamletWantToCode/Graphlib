import torch
from ..base import BaseDataLoader
from .Alchemy_dataset import TencentAlchemyDataset


class AlchemyDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        if training:
            mode = 'dev'
        else:
            mode = 'valid'
        
        transform = None
        self.dataset = TencentAlchemyDataset(data_dir, mode, transform=transform)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
