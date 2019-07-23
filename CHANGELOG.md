# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- `samplers`: various data sampling methods for pytorch data loader
- `transform`: data transformation method for molecular graph
- `trainer`: providing constraints, initializers, regularizers ......
- `test.py`: auto-read the best model's track file; train model on the whole training dataset before prediction

## [0.0.1] - 2019-07-23
### Added
- `base`: base classes for *dataloader*, *trainer* and *model*
- `nn`: layers for the state-of-the-art graph neural networks
- `model`: graph neural network architecture
- `trainer`: high-level training interface which abstracts away the training loop
- `logger`: record the training process, monitering the change of error and loss
- `utils`: various utility function for logger

### Changed
- `TencentAlchemyDataset`: use `torchvision.dataset.DatasetFolder` to read *sdf* file from the *raw* folder