from DistributedSim.trainer import LocalTrainer

from DistributedSim.strategy.sparta import SPARTAStrategy

from DistributedSim.models.nanogpt import GPT, GPTConfig
from DistributedSim.dataset.nanogpt.dataset import get_dataset

import torch
from functools import partial

def main():
  train_dataset, vocab_size = get_dataset('shakespeare', block_size=1024, storage_device='cpu', compute_device='mps', start_pc=0.0, end_pc=0.9)
  val_dataset, vocab_size = get_dataset('shakespeare', block_size=1024, storage_device='cpu', compute_device='mps', start_pc=0.9, end_pc=1.0)

  model = GPT(GPTConfig(n_layer=12, n_head=12, n_embd=768, vocab_size=vocab_size))

  trainer = LocalTrainer(
    model, 
    train_dataset, 
    val_dataset, 
  )

  # optim = torch.optim.AdamW(model.parameters(), lr=0.001)
  optim = partial(torch.optim.AdamW, lr=0.001)
  strategy = SPARTAStrategy(
    optim=optim
  )

  trainer.fit(
    num_epochs=1,
    strategy=strategy,
    num_nodes=4,
    device='mps'
  )

if __name__ == '__main__':
  main()