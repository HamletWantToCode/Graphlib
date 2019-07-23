# Graphlib

Implementation of latest graph neural network model using Pytorch and torch_geometric.

## Structure

### base
- `BaseDataLoader`
- `BaseModel`
- `BaseTrainer`

### nn
Various neural network layer for graph neural network

- `SGC_LL`: **Adaptive Graph Convolutional Neural Networks.**
*Ruoyu Li, Sheng Wang, Feiyun Zhu, Junzhou Huang.* AAAI 2018. [paper](https://arxiv.org/pdf/1801.03226.pdf)

- `graph_max_pool`: same as above

### data loader
Customized data loaders for various datasets

-  `AlchemyDataLoader`: Tecent *Alchemy* dataset

### model
The state-of-the-art graph neural network model

- `AGCN`: **Adaptive Graph Convolutional Neural Networks.**
*Ruoyu Li, Sheng Wang, Feiyun Zhu, Junzhou Huang.* AAAI 2018. [paper](https://arxiv.org/pdf/1801.03226.pdf)

- `MPNN`: **Neural Message Passing for Quantum Chemistry.**
*Gilmer, Justin and Schoenholz, Samuel S and Riley, Patrick F and Vinyals, Oriol and Dahl, George E.* 2017. [paper](https://arxiv.org/pdf/1704.01212.pdf)

### trainer
Customized trainer for training the model and save the training log

- `Trainer`

## Installation

## Usage

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## TODOs
- [ ] Implement common feature transformation for molecular graph
- [ ] Multi-GPU support

## License
[MIT](https://choosealicense.com/licenses/mit/)