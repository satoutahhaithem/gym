# Distributed Sim

A framework for simulated distributed training methods.
Instead of training with multiple ranks, we simulate the distributed training process by running multiple nodes on a single machine.

## Supported Devices

- CPU
- CUDA
- MPS (CPU-bound for copy operations, see [here](https://github.com/pytorch/pytorch/issues/141287))

## Supported Methods

- AllReduce (Equivalent to PyTorch DistributedDataParallel)
- SPARTA - Sparse Parameter Averaging
- [DiLoCo](https://arxiv.org/abs/2311.08105)
- [DeMo](https://arxiv.org/abs/2411.19870)

## Repo Structure

- `SimBuilder`: Abstract class for building the simulation environment. `SimBuilder` will spawn multiple `TrainNode` instances, and connect them depending on the method.
- `TrainNode`: A single node (rank) running its own training loop. At each train step, instead of calling `optim.step()`, it calls `gradient_strategy.step()`.
- `GradientStrategy`: Abstract class for a gradient strategy, which both defines how the nodes communicate with each other and how the model weights are updated. Typically, a gradient strategy will include an optimizer as well as a communication step.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
cd experiments
python nanogpt_diloco.py --dataset shakespeare --char --num_nodes 2
```