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

## Installation

### Basic Installation
Install with core dependencies only:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ exogym
```

### Installation with Optional Features

Optional feature flags allowed are:

```bash
wandb,gpt,demo,examples,all,dev
```

For example, `pip install exogym[demo]`

### Development Installation

To install for development:
```bash
git clone https://github.com/exo-explore/gym.git exogym
cd exogym
pip install -e ".[dev]"
```

## Simulation Explained

In this simulation, the model is learning to predict which digit (from 0 to 9) is written in an image.

The program is using the famous MNIST dataset, which is a large collection of 28x28 pixel images of handwritten digits.

Here's how it works:

The model is shown an image of a handwritten digit.
It makes a prediction about which digit it is (e.g., "I think this is a 7").
It then compares its prediction to the actual label of the image.
The loss value you see in the logs is a measure of how wrong the model's prediction was. A high loss means the prediction was very wrong, and a low loss means it was close to correct.
The goal of the training is to adjust the model's internal parameters to make the loss as low as possible. As you can see from the logs, the train_loss is generally decreasing over time, which means the model is getting better at correctly identifying the handwritten digits.

### The Relationship Between Training and Prediction

You've hit on a key distinction. You are correct, DiLoCo is used for the training of the model, not for the prediction itself.

Here's the relationship between the two:

**The Goal (Prediction):** The ultimate objective is to create a model that can accurately predict handwritten digits. This is the task we want the final, trained model to perform.

**The Method (DiLoCo):** DiLoCo is the strategy we use to achieve that goal. Think of it as the recipe for training. Since we are using two nodes (like two chefs working together), we need a strategy to coordinate their work. DiLoCo is a specific set of rules for how these two nodes should:

1.  Learn from their own batches of data.
2.  Communicate what they've learned to each other efficiently.
3.  Combine their knowledge to create a better, unified model.

So, we use the DiLoCo training strategy to produce a digit prediction model. The same prediction goal could be achieved using other training strategies like SPARTA, SimpleReduceStrategy, FedAvg, or DeMo, but the training process itself—the speed, the communication between nodes, and the final accuracy—would be different.

Here's a brief overview of the other strategies:

*   **FedAvg (Federated Averaging):** This is a classic federated learning algorithm. Each node trains the model on its own data for a few steps, and then the models from all the nodes are averaged together to create a new global model. This process is repeated until the model converges.
*   **DeMo (Decoupled Momentum):** This is a more advanced strategy that uses a technique called decoupled momentum to improve the training process. It's designed to be more efficient than traditional methods like `SimpleReduce`.

## How to Run the Simulation

To run the simulation, you can execute the following command in your terminal:

```bash
python3 example/mnist.py
```

This will start the training process using the settings defined in the `example/mnist.py` script.

## Benchmarking Results

### Machine Specifications

The benchmarks were run on the following machine:

*   **CPU:** Intel(R) Xeon(R) CPU E5-1620 v3 @ 3.50GHz (8 cores)
*   **GPU:** Quadro RTX 6000
*   **RAM:** 31Gi

### Results

Here is a comparison of the results from the five training strategies:

| Strategy | Final Loss | Training Time | Iterations/sec | Final LR |
| :--- | :--- | :--- | :--- | :--- |
| **SimpleReduce (AllReduce)** | 0.0601 | 3 min 29s | 2.82it/s | 0.000030 |
| **SPARTA** | 0.0493 | 3 min 30s | 2.80it/s | 0.000100 |
| **DiLoCo** | 0.0197 | 3 min 9s | 3.11it/s | 0.000030 |
| **FedAvg** | 0.0193 | 3 min 9s | 3.11it/s | 0.000000 |
| **DeMo** | 0.0309 | 3 min 45s | 2.62it/s | 0.000000 |

### Understanding the Benchmarking Results

*   **Final Loss:** This is the most important metric for determining the accuracy of the trained model. A lower final loss means the model is better at predicting the correct digit.
*   **Training Time:** This is the total time it took to train the model. A shorter training time is better.
*   **Iterations/sec:** This metric shows how many training iterations (or steps) were completed per second. A higher number is better, as it indicates a more efficient training process.
*   **Final LR:** This is the final learning rate of the optimizer at the end of the training process. A value of 0.000000 indicates that no learning rate scheduler was used, and the learning rate did not change from its initial value.

### Analysis of the Results

*   **`FedAvg` and `DiLoCo`** are the top-performing strategies in this benchmark. They both achieve a very low final loss in a short amount of time. `FedAvg` has a slightly lower final loss, but the difference is negligible.
*   **`DeMo`** is a solid performer, but it is slightly slower and has a higher final loss than `FedAvg` and `DiLoCo`.
*   **`SPARTA`** is a bit slower than the top performers and has a higher final loss.
*   **`SimpleReduce`** is the slowest and has the highest final loss, which is expected as it is the most communication-intensive and serves as a baseline.

In conclusion, for this specific task and dataset, `FedAvg` and `DiLoCo` are the best choices for achieving high accuracy in a short amount of time.

## Codebase Structure

- `Trainer`: Builds simulation environment. `Trainer` will spawn multiple `TrainNode` instances, connect them together, and starts the training run.
- `TrainNode`: A single node (rank) running its own training loop. At each train step, instead of calling `optim.step()`, it calls `strategy.step()`.
- `Strategy`: Abstract class for an optimization strategy, which both defines **how the nodes communicate** with each other and **how model weights are updated**. Typically, a gradient strategy will include an optimizer as well as a communication step. Sometimes (eg. DeMo), the optimizer step is comingled with the communication.

## Technical Details

EXO Gym uses pytorch multiprocessing to spawn a subprocess per-node, which are able to communicate with each other using regular operations such as `all_reduce`.

### Model

The model is expected in a form that takes a `batch` (the same format as `dataset` outputs), and returns a scalar loss over the entire batch. This ensures the model is agnostic to the format of the data (eg. masked LM training doesn't have a clear `x`/`y` split).

### Dataset

Recall that when we call `trainer.fit()`, $K$ subprocesses are spawned to handle each of the virtual workers. There are two options for creating dataset:

#### PyTorch `Dataset`

Instantiate a single `Dataset`. The `dataset` object is passed to every subprocess, and a `DistributedSampler` will be used to select which datapoints are sampled per-node (to ensure each datapoint is only used once by each node). If the dataset is entirely loaded into memory, this memory will be duplicated per-node - be careful not to run out of memory! If the dataset is larger, it should be lazily loaded.

#### `dataset_factory` function

In place of the dataset object, pass a function with the following signature:

```python
def dataset_factory(rank: int, num_nodes: int, train_dataset: bool) -> torch.utils.data.Dataset
```

This will be called within each rank to build the dataset. Instead of each node storing the whole dataset and subsampling datapoints, each node only loads the necessary datapoints.