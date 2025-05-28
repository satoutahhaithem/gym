from DistributedSim.trainer import LocalTrainer

from DistributedSim.strategy.sparta import SPARTAStrategy
from DistributedSim.strategy.diloco import DiLoCoStrategy
from DistributedSim.strategy.sparta_diloco import SPARTADiLoCoStrategy
from DistributedSim.strategy.federated_averaging import FedAvgStrategy
from DistributedSim.strategy.demo import DeMoStrategy

from DistributedSim.example.nanogpt.nanogpt import GPT, GPTConfig
from DistributedSim.example.nanogpt.dataset import get_dataset
from DistributedSim.strategy.optim import OptimSpec

import torch

def main():
  train_dataset, vocab_size = get_dataset('shakespeare', block_size=1024, device='cpu', start_pc=0.0, end_pc=0.9)
  val_dataset, vocab_size = get_dataset('shakespeare', block_size=1024, device='cpu', start_pc=0.9, end_pc=1.0)

  model = GPT(GPTConfig(n_layer=12, n_head=12, n_embd=768, vocab_size=vocab_size))

  trainer = LocalTrainer(
    model, 
    train_dataset, 
    val_dataset, 
  )

  optim = OptimSpec(
    torch.optim.AdamW,
    lr=0.001
  )

  strategy = FedAvgStrategy(
  # strategy = SPARTAStrategy(
  # strategy = SPARTADiLoCoStrategy(
  # strategy = DiLoCoStrategy(
    inner_optim=optim,
    H=10,
    lr_scheduler='lambda_cosine',
    lr_scheduler_kwargs={
      'warmup_steps': 100,
      'cosine_anneal': True
    }
  )

  # strategy = SPARTAStrategy(
  #   optim_spec=optim,
  #   lr_scheduler='lambda_cosine',
  #   lr_scheduler_kwargs={
  #     'warmup_steps': 100,
  #     'cosine_anneal': True
  #   }
  # )

  trainer.fit(
    num_epochs=1,
    strategy=strategy,
    num_nodes=4,
    device='mps',
    batch_size=1,
    minibatch_size=1,
    val_size=1,
    wandb_project='diloco-test',
    wandb_name='run_1'
  )

if __name__ == '__main__':
  main()