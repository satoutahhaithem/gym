import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from exogym.train_node import TrainNode
from exogym.strategy import Strategy

import os
from abc import abstractmethod
import copy
from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Union, Callable
from collections import OrderedDict

def print_dataset_size(dataset: torch.utils.data.Dataset):
    import pickle
    import io

    buffer = io.BytesIO()
    pickle.dump(dataset, buffer, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Dataset size: {buffer.tell() // 1024 // 1024} MB")


@dataclass
class TrainingConfig:
    """Configuration class that holds all training parameters for serialization."""

    model: torch.nn.Module
    train_dataset: Union[
        torch.utils.data.Dataset, Callable[[int, int, bool], torch.utils.data.Dataset]
    ]
    val_dataset: Union[
        torch.utils.data.Dataset, Callable[[int, int, bool], torch.utils.data.Dataset]
    ]
    strategy: Strategy
    num_epochs: int
    num_nodes: int
    max_steps: Optional[int] = None
    device: Optional[str] = None
    devices: Optional[List[int]] = None
    batch_size: int = 16
    minibatch_size: int = 16
    shuffle: bool = True
    val_size: int = 64
    val_interval: int = 100
    autocast: bool = False
    checkpoint_interval: int = 100
    trainer_class: type = None
    kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


def _worker(rank: int, config: TrainingConfig, result_queue: mp.Queue):
    """
    Entry point executed in every child process.
    This function is importable as exogym.trainer._worker, making it notebook-safe.
    """
    # Create trainer instance in the worker process
    trainer = config.trainer_class(
        model=config.model,
        train_dataset=config.train_dataset,
        val_dataset=config.val_dataset,
        **config.kwargs,
    )

    # Set all the configuration parameters
    trainer.num_epochs = config.num_epochs
    trainer.max_steps = config.max_steps
    trainer.strategy = config.strategy
    trainer.num_nodes = config.num_nodes
    trainer.device = config.device
    trainer.devices = config.devices
    trainer.batch_size = config.batch_size
    trainer.minibatch_size = config.minibatch_size
    trainer.shuffle = config.shuffle
    trainer.val_size = config.val_size
    trainer.val_interval = config.val_interval
    trainer.autocast = config.autocast
    trainer.checkpoint_interval = config.checkpoint_interval

    # Run the training process and get the final model state dict
    final_model_state = trainer._fit_process(rank)

    # Move tensors to CPU and detach to avoid CUDA serialization issues
    cpu_state_dict = OrderedDict()
    for key, tensor in final_model_state.items():
        cpu_state_dict[key] = tensor.detach().cpu()

    # Put the result in the queue
    result_queue.put((rank, cpu_state_dict))

def _average_model_states(model_states: Dict[int, OrderedDict]) -> OrderedDict:
    """Average model state dictionaries from multiple processes."""
    if not model_states:
        return None

    # Get the first state dict as template
    averaged_state = OrderedDict()
    first_state = list(model_states.values())[0]

    # Average each parameter
    for param_name in first_state.keys():
        # Stack all versions of this parameter
        param_stack = torch.stack(
            [state[param_name] for state in model_states.values()]
        )
        # Average them
        if param_stack.dtype.is_floating_point or param_stack.dtype.is_complex:
            averaged_state[param_name] = torch.mean(param_stack, dim=0)
        else:
            # For integer types, cast to float for mean, then cast back.
            averaged_state[param_name] = torch.mean(param_stack.float(), dim=0).to(
                param_stack.dtype
            )

    return averaged_state


class Trainer:
    """
    Trainer is used to train a model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Union[
            torch.utils.data.Dataset,
            Callable[[int, int, bool], torch.utils.data.Dataset],
        ],
        val_dataset: Union[
            torch.utils.data.Dataset,
            Callable[[int, int, bool], torch.utils.data.Dataset],
        ],
        start_port: Optional[int] = None,
        **kwargs,
    ):
        self.model_orig = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.kwargs = kwargs
        self.port = start_port if start_port is not None else 12355

    def fit(
        self,
        num_epochs: int,
        strategy: Strategy,
        num_nodes: int,
        max_steps: int = None,
        device: str = None,
        devices: List[int] = None,
        batch_size: int = 16,
        minibatch_size: int = 16,
        shuffle: bool = True,
        val_size: int = 64,
        val_interval: int = 100,
        autocast: bool = False,
        checkpoint_interval: int = 100,
        **kwargs,
    ):
        """
        Train the model. For single process training (num_nodes=1), runs directly.
        For multi-process training, delegates to launch_ddp for notebook safety.
        Returns the final trained model (averaged across nodes for multi-node training).
        """
        # Store parameters
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.strategy = strategy
        self.num_nodes = num_nodes
        self.device = device
        self.devices = devices
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.shuffle = shuffle
        self.val_size = val_size
        self.val_interval = val_interval
        self.autocast = autocast
        self.checkpoint_interval = checkpoint_interval

        assert self.val_size // self.batch_size > 0, "val_size must be geq batch_size"

        if hasattr(self, "kwargs"):
            self.kwargs.update(kwargs)
        else:
            self.kwargs = kwargs

        self.port += 1

        # Move a *copy* of the model to CPU so that pickling for mp.spawn does not attempt to share GPU storage.
        cpu_model = copy.deepcopy(self.model_orig).cpu()

        config = TrainingConfig(
            model=cpu_model,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            strategy=strategy,
            num_epochs=num_epochs,
            num_nodes=num_nodes,
            max_steps=max_steps,
            device=device,
            devices=devices,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            shuffle=shuffle,
            val_size=val_size,
            val_interval=val_interval,
            autocast=autocast,
            checkpoint_interval=checkpoint_interval,
            trainer_class=self.__class__,
            kwargs=self.kwargs,
        )
        
        # Create a manager and queue for collecting results
        manager = mp.Manager()
        result_queue = manager.Queue()

        # Use mp.spawn with the result queue
        mp.spawn(
            _worker,
            args=(config, result_queue),
            nprocs=config.num_nodes,
            start_method="spawn",
            join=True,
        )

        # Collect results
        model_states = {}
        for _ in range(config.num_nodes):
            rank, state_dict = result_queue.get()
            model_states[rank] = state_dict

        # Average the models
        averaged_state_dict = _average_model_states(model_states)

        if averaged_state_dict is not None:
            # Create a copy of the original model and load the averaged state
            final_model = copy.deepcopy(self.model_orig)
            final_model.load_state_dict(averaged_state_dict)
            return final_model
        else:
            return None

    def _fit_process(self, rank):
        """
        The core training logic that runs in each process.
        Renamed from _fit and removed the spawn call.
        Returns the final model state dict.
        """
        self.rank = rank

        self._build_connection()

        self.model = copy.deepcopy(self.model_orig).to(self.device)

        self.strategy = copy.deepcopy(self.strategy)
        self.strategy._init_node(self.model, self.rank, self.num_nodes)

        # Handle dataset factory vs direct dataset for sampler creation
        if callable(self.train_dataset):
            # For dataset factory, we don't need a distributed sampler
            # since the factory should return the appropriate subset for this rank
            self.sampler = None
        else:
            # For direct dataset, use DistributedSampler as before
            self.sampler = torch.utils.data.DistributedSampler(
                self.train_dataset,
                num_replicas=self.num_nodes,
                rank=self.rank,
                shuffle=self.shuffle,
            )

        sim = TrainNode(
            self.model,
            self.train_dataset,
            self.sampler,
            self.val_dataset,
            self.strategy,
            self.device,
            self.rank,
            self.num_nodes,
            num_epochs=self.num_epochs,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            minibatch_size=self.minibatch_size,
            val_size=self.val_size,
            val_interval=self.val_interval,
            checkpoint_interval=self.checkpoint_interval,
            autocast=self.autocast,
            **self.kwargs,
        )

        final_state_dict = sim.train()

        self._process_cleanup()

        return final_state_dict

    @abstractmethod
    def _build_connection(self):
        raise NotImplementedError

    def _process_cleanup(self):
        dist.destroy_process_group()


class LocalTrainer(Trainer):
    def _build_connection(self):
        """
        This is the default callback for setting up pytorch distributed connections.
        All ranks are assumed to be on the same machine, and device is defaulted to cpu.
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.port)

        if self.device == "" or self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # initialize the process group
        if self.device == "cuda":
            # If we haven't specified devices, use all devices.
            if self.devices is None:
                self.devices = range(torch.cuda.device_count())

            dist.init_process_group(
                "nccl" if len(self.devices) == self.num_nodes else "gloo",
                rank=self.rank,
                world_size=self.num_nodes,
            )
            self.device = torch.device(
                f"cuda:{self.devices[self.rank % len(self.devices)]}"
            )
            torch.cuda.set_device(self.device)
        elif self.device == "cpu":
            dist.init_process_group("gloo", rank=self.rank, world_size=self.num_nodes)
            self.device = torch.device("cpu")
        elif self.device == "mps":
            dist.init_process_group("gloo", rank=self.rank, world_size=self.num_nodes)
            self.device = torch.device("mps")
        else:
            raise ValueError(f"Invalid device type: {self.device}")

        print(f"Rank {self.rank} using device {self.device}")