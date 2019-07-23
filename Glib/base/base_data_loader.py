import numpy as np
from torch_geometric.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler



class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, **kwargs):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split, **kwargs)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'follow_batch': []
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split, stratify=None):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        if stratify is None:
            print('randomly sample from the indices')
            _valid_idx = idx_full[0:len_valid]
            _train_idx = np.delete(idx_full, np.arange(0, len_valid))
        else:
            from sklearn.model_selection import train_test_split
            
            print('using stratified sampling method')
            _train_idx, _valid_idx = train_test_split(idx_full, test_size=len_valid, stratify=stratify)

        train_idx = [int(x) for x in _train_idx]
        valid_idx = [int(x) for x in _valid_idx]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
