import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from typing import Optional, Callable, Type
import os

from sim_config import *
from gradient_strategy import *

class TrainNode:
    def __init__(self, 
                 config: SimConfig,
#                  model_class: Type[torch.nn.Module],
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader,
                 device: torch.device,
                 rank: int):
        self.config = config

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.rank = rank

        self.model = self.config.model_class(**self.config.model_kwargs).to(self.device) # TODO: model kwargs
        
        # def print_grad(grad):
        #     print(grad)
        # self.model.encoder_layer.self_attn.in_proj_weight.register_hook(print_grad)

        ## Ensure all process models share the same params
        for _, param in self.model.named_parameters():
            dist.broadcast(param.data, src=0)

        self.criterion = self.config.criterion_class(**self.config.criterion_kwargs)

        self.gradient_strategy = self.config.gradient_class(self.model, self.config)

        # for _, param in self.model.named_parameters():
        #     print(f'Process {self.rank} params {param}')
        #     break

    def train_epoch(self):
        self.model.train()

        if int(os.environ['VERBOSITY']) >= 3:
            print(f'Process {self.rank} train')

        loss_sum = torch.tensor(0, device=self.device)

        train_loss_list = []

        for i, batch in enumerate(self.train_dataloader, 0):
            self.gradient_strategy.zero_grad()

            X, y = batch
            yhat = self.model.forward(X)

            loss = self.criterion(yhat, y)
            
            loss.backward()
            dist.barrier()

            self.gradient_strategy.step()

            loss_sum = torch.add(loss_sum, loss)
            train_loss_list.append(loss.item())

            if i % 100 == 0:
                dist.barrier()
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()

                if self.rank == 0:
                    print(f'step {i}/{len(self.train_dataloader)}: ' + \
                        f'loss {(loss.item() / self.config.batch_size):.6f}')

        # dist.barrier()
        # dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)

        return train_loss_list

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
        val_losses = []
        train_losses = []

        for epoch in range(epochs):
            self.train_dataloader.sampler.set_epoch(epoch)

            val_loss, val_accuracy = self.val_epoch()

            if self.rank == 0:
                print(val_loss / (len(self.val_dataloader) * self.config.batch_size), val_accuracy)
            
            val_losses.append((val_loss, val_accuracy))

            train_loss_list = self.train_epoch()
            train_losses += train_loss_list

        return val_losses, train_losses
