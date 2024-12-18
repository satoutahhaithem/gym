import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from typing import Optional, Callable, Type
import os

from sim_config import *

class TrainNode:
    def __init__(self, 
                 config: SimConfig,
#                  model_class: Type[torch.nn.Module],
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader,
#                  optimizer: Type[torch.optim.Optimizer],
                 device: torch.device,
                 rank: int):
        self.config = config

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.rank = rank

        self.model = self.config.model_class().to(self.device) # TODO: model kwargs
        ## Ensure all process models share the same params
        for _, param in self.model.named_parameters():
            dist.broadcast(param.data, src=0)

        self.optimizer = self.config.optimizer_class(self.model.parameters(),
                                                     **self.config.optimizer_kwargs)

        self.criterion = self.config.criterion_class(**self.config.criterion_kwargs)

        # for _, param in self.model.named_parameters():
        #     print(f'Process {self.rank} params {param}')
        #     break

    def train_epoch(self):
        self.model.train()

        if int(os.environ['VERBOSITY']) >= 3:
            print(f'Process {self.rank} train')

        loss_sum = torch.tensor(0, device=self.device)

        for i, batch in enumerate(self.train_dataloader, 0):
            self.optimizer.zero_grad()

            X, y = batch
            yhat = self.model.forward(X)

            loss = self.criterion(yhat, y)

            # TODO: AllReduce is currently doing the default - summing gradients (high bandwidth cost) 
            loss.backward()
            dist.barrier()

            for param in self.model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()

            self.optimizer.step()

            for name, param in self.model.named_parameters():
#                 print(f'\nProcess {self.rank} {name} gradient {param.grad}')
                break

            loss_sum = torch.add(loss_sum, loss)

            if i % 100 == 0:
                dist.barrier()
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()

                if self.rank == 0:
                    print(f'step {i}/{len(self.train_dataloader)}: ' + \
                        f'loss {(loss.item() / self.config.batch_size):.6f}')

        # print(f'\n Process {self.rank} train loss {loss_sum}')

        dist.barrier()
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)

        return loss_sum

    def val_epoch(self):
        self.model.eval()

        loss_sum = torch.tensor(0, device=self.device)
        correct = torch.tensor(0, device=self.device)
        total = torch.tensor(0, device=self.device)

        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader, 0):
                X, y = batch
                yhat = self.model.forward(X)

                loss = self.criterion(yhat, y)
                loss_sum = torch.add(loss_sum, loss)

                # Calculate accuracy
                _, predicted = torch.max(yhat, dim=1)  # Get the index of the max logit
                correct += (predicted == y).sum()      # Count correct predictions
                total += y.size(0)                     # Total samples

        ## Sum losses and correct/total counts across nodes
        dist.barrier()
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

        accuracy = 100.0 * correct.item() / total.item()

        return loss_sum.item(), accuracy


    def train(self, epochs=10):
        for epoch in range(epochs):
            val_loss, val_accuracy = self.val_epoch()

            if self.rank == 0:
                print(val_loss / (len(self.val_dataloader) * self.config.batch_size), val_accuracy)

            train_loss = self.train_epoch()
