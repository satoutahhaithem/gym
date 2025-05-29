# EXO Gym

Open source framework for simulated distributed training methods.
Instead of training with multiple ranks, we simulate the distributed training process by running multiple nodes on a single machine.

## Supported Devices

- CPU
- CUDA
- MPS (CPU-bound for copy operations, see [here](https://github.com/pytorch/pytorch/issues/141287))

## Supported Methods

- AllReduce (Equivalent to PyTorch [DDP](https://arxiv.org/abs/2006.15704))
- [FedAvg](https://arxiv.org/abs/2311.08105)
- [DiLoCo](https://arxiv.org/abs/2311.08105)
- [SPARTA](https://openreview.net/forum?id=stFPf3gzq1)
- [DeMo](https://arxiv.org/abs/2411.19870)

## Example Usage

```python
from exogym import LocalTrainer
from exogym.strategy import DiLoCoStrategy

train_dataset, val_dataset = ...
model = ...

trainer = LocalTrainer(model, train_dataset, val_dataset)

strategy = DiLoCoStrategy(
  inner_optim='adam',
  H=100
)

trainer.fit(
  strategy=strategy,
  num_nodes=4,
  device='mps'
)
```

## Installation

### Basic Installation
Install with core dependencies only:
```bash
pip install exogym
```

### Installation with Optional Features

For experiment tracking with Weights & Biases:
```bash
pip install exogym[wandb]
```

For S3 dataset loading:
```bash
pip install exogym[s3]
```

For DeMo strategy support:
```bash
pip install exogym[demo]
```

For running examples:
```bash
pip install exogym[examples]
```

For all optional features:
```bash
pip install exogym[all]
```

For development:
```bash
pip install exogym[dev]
```

### Development Installation

To install for development:
```bash
git clone https://github.com/MattyAB/DistributedSim.git
cd DistributedSim
pip install -e .[dev]
```

## Codebase Structure

- `Trainer`: Builds simulation environment. `Trainer` will spawn multiple `TrainNode` instances, connect them together, and starts the training run.
- `TrainNode`: A single node (rank) running its own training loop. At each train step, instead of calling `optim.step()`, it calls `strategy.step()`.
- `Strategy`: Abstract class for an optimization strategy, which both defines **how the nodes communicate** with each other and **how model weights are updated**. Typically, a gradient strategy will include an optimizer as well as a communication step. Sometimes (eg. DeMo), the optimizer step is comingled with the communication.

## Technical Details

EXO Gym uses pytorch multiprocessing to a subprocess per-node, which are able to communicate with each other using regular operations such as `all_reduce`.

### Model

The model is expected in a form that takes a `batch` (the same format as `dataset` outputs), and returns a scalar loss over the entire batch. This ensures the model is agnostic to the format of the data (eg. masked LM training doesn't have a clear `x`/`y` split).

### Dataset

Recall that when we call `trainer.fit()`, $K$ subprocesses are spawned to handle each of the virtual workers. The `dataset` object is passed to every subprocess, and a `DistributedSampler` will be used to select indices per-node. If the dataset is entirely loaded into memory, this memory will be duplicated per-node - be careful not to run out of memory! If the dataset is larger, it should be lazily loaded.

<!-- For further information, see individual pages on:

- [Dataset](./docs/dataset.md) -->