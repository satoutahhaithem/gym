import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np
import zipfile

import os
import copy

from .strategy.strategy import Strategy
from .logger import Logger, WandbLogger
from .strategy.communicate import *
from .utils import LogModule

# change to two-space indent instead of four-space (which is what it is at the moment)

class TrainNode(LogModule):
    '''
    Single node of distributed training process. Should be the same regardless of rank topology/architecture.
    '''
    def __init__(self, 
                 model: torch.nn.Module,
                 train_dataset: torch.utils.data.Dataset,
                 train_sampler: torch.utils.data.Sampler,
                 val_dataset: torch.utils.data.Dataset,
                 strategy: Strategy,
                 device: torch.device,
                 rank: int,
                 num_nodes: int,
                 num_epochs: int,
                 max_steps: int = None,
                 batch_size: int = 16, 
                 minibatch_size: int = 16,
                 val_size: int = 64, 
                 val_interval: int = 100,
                 checkpoint_interval: int = 100,
                 autocast: bool = False,
                 **kwargs):

        self.model = model
        self.train_dataset = train_dataset
        self.train_sampler = train_sampler
        self.val_dataset = val_dataset
        self.strategy = strategy
        self.device = device
        self.rank = rank
        self.num_nodes = num_nodes
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.val_size = val_size
        self.val_interval = val_interval
        self.autocast = autocast
        self.checkpoint_interval = checkpoint_interval

        self.kwargs = kwargs

        self.build_dataloaders()

        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        ## Ensure all process models share the same params
        if self.num_nodes > 1:
            for _, param in self.model.named_parameters():
                broadcast(param.data, src=0)

        self.local_step = 0
        self.epoch = 0
        
        # Attempt to load checkpoint before starting training
        self._load_checkpoint()

    def build_dataloaders(self):
        """
        Builds dataloaders.
        """
        self.train_dataloader = DataLoader(self.train_dataset, 
                          batch_size=self.minibatch_size,
                          sampler=self.train_sampler)

        self.val_dataloader = DataLoader(self.val_dataset, 
                          batch_size=self.minibatch_size,
                          shuffle=True)

        self.train_data_iter = iter(self.train_dataloader)
        self.val_data_iter = iter(self.val_dataloader)

    def _get_batch(self, eval=False):
        import time
        start_time = time.time()
        
        if not eval or self.val_data_iter is None:
            try:
                batch = next(self.train_data_iter)
            except StopIteration:
                self.epoch += 1
                self.train_data_iter = iter(self.train_dataloader)
                batch = next(self.train_data_iter)
        else:
            try:
                batch = next(self.val_data_iter)
            except StopIteration:
                self.val_data_iter = iter(self.val_dataloader)
                batch = next(self.val_data_iter)

        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = tuple(x.to(self.device) for x in batch)
        else:
            batch = batch.to(self.device)
        
        end_time = time.time()
        # print(f"Batch collection time: {end_time - start_time:.4f} seconds")
        
        return batch

    def _train_step(self):
        self.strategy.zero_grad()
        
        for i in range(self.batch_size // self.minibatch_size):
            minibatch = self._get_batch()

            ## TODO: Do we want this?
            if self.autocast:
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    loss = self.model(minibatch)
            else:
                loss = self.model(minibatch)

            loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad /= (self.batch_size / self.minibatch_size)
        
        self.strategy.step()

        if self.rank == 0:
            self.logger.log_train(loss=loss.item())

        if self.checkpoint_interval and self.local_step % self.checkpoint_interval == 0:
            self._save_checkpoint()

    def _evaluate(self):
        model_clone = copy.deepcopy(self.model)

        for name, param in model_clone.named_parameters():
            all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data = param.data / dist.get_world_size()

        if self.rank == 0:
            # For rank 0, we will calculate the local loss
            this_model = self.model

        if self.rank == 1:
            # For rank 1, we want to calculate the average model loss
            this_model = model_clone


        if self.rank == 0 or self.rank == 1:
            this_model.eval()
            
            loss_total = 0

            with torch.no_grad():
                for _ in range(int(self.val_size / self.batch_size)):

                    for i in range(self.batch_size // self.minibatch_size):
                        minibatch = self._get_batch(eval=True)

                        if self.autocast:
                            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                                ## TODO: Fix
                                loss = this_model(minibatch)
                        else:
                            loss = this_model(minibatch)

                        loss_total += loss.item() / (self.batch_size // self.minibatch_size)

        # Rank 0 logs the local evaluation.
        if self.rank == 0:
            self.logger.log_loss(loss=loss_total / int(self.val_size / self.batch_size), 
                                    name='val_local')

        # Broadcast the global loss from rank 1 to all ranks.
        if self.num_nodes > 1:
            # All ranks create a dummy tensor to participate.
            global_loss_tensor = torch.empty(1, device=next(self.model.parameters()).device)
            if self.rank == 1:
                global_loss_tensor[0] = loss_total / int(self.val_size / self.batch_size)
            broadcast(global_loss_tensor, src=1)

            # Only rank 0 logs the global evaluation.
            if self.rank == 0:
                global_loss = global_loss_tensor.item()
                self.logger.log_loss(loss=global_loss, name='global')

        del model_clone


    def _save_checkpoint(self):
        return ## TODO
        print(self.config.save_dir, self.config.wandb_project, self.config.wandb_name, self.rank)
        save_path_dir = os.path.join(self.config.save_dir,
                                 self.config.wandb_project if self.config.wandb_project else 'unnamed',
                                 self.config.wandb_name if self.config.wandb_name else 'unnamed',
                                 str(self.rank))
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir, exist_ok=True)

        filename = f"{self.local_step}.pt"
        full_save_path = os.path.join(save_path_dir, filename)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.strategy.optim.state_dict(),
            'local_step': self.local_step,
            'epoch': self.epoch,
            'rng_state': torch.get_rng_state(),
        }
        if self.strategy.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.strategy.scheduler.state_dict()
        
        if self.device.type == 'cuda':
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()

        try:
            torch.save(checkpoint, full_save_path)
            print(f"Rank {self.rank} saved checkpoint to {full_save_path} at step {self.local_step}")
            self._delete_other_checkpoints(save_path_dir, full_save_path)
        except OSError as e:
            print(f"Rank {self.rank}: Failed to save checkpoint {full_save_path} due to OSError: {e}. Attempting to delete oldest checkpoint and retry.")
            
            oldest_step = float('inf')
            oldest_checkpoint_file = None
            # Ensure save_path_dir exists before listing its contents, though it should have been created.
            if os.path.exists(save_path_dir):
                for f_name in os.listdir(save_path_dir):
                    if f_name.endswith('.pt'):
                        try:
                            # Checkpoints are named as {step_num}.pt
                            step_num = int(f_name.split('.')[0])
                            if step_num < oldest_step:
                                oldest_step = step_num
                                oldest_checkpoint_file = f_name
                        except ValueError:
                            # Skip files not matching the expected N.pt pattern
                            continue
            
            if oldest_checkpoint_file:
                oldest_checkpoint_path = os.path.join(save_path_dir, oldest_checkpoint_file)
                try:
                    os.remove(oldest_checkpoint_path)
                    print(f"Rank {self.rank}: Deleted oldest checkpoint {oldest_checkpoint_path} to free space.")
                    
                    # Retry saving the current checkpoint
                    try:
                        torch.save(checkpoint, full_save_path)
                        print(f"Rank {self.rank}: Successfully saved checkpoint {full_save_path} after deleting oldest.")
                        self._delete_other_checkpoints(save_path_dir, full_save_path)
                    except OSError as e2:
                        print(f"Rank {self.rank}: Still failed to save checkpoint {full_save_path} after deleting oldest: {e2}. Giving up.")
                        raise # Re-raise the second error, as we couldn't save even after cleanup
                except OSError as del_e:
                    print(f"Rank {self.rank}: Failed to delete oldest checkpoint {oldest_checkpoint_path}: {del_e}. Original save error will be raised.")
                    raise e # Re-raise the original save error, as cleanup failed
            else:
                print(f"Rank {self.rank}: No old checkpoints found to delete in {save_path_dir}. Original save error will be raised.")
                raise e # Re-raise the original save error, as no space could be freed

    def _delete_other_checkpoints(self, save_path_dir: str, current_checkpoint_full_path: str):
        return ## TODO
        if not os.path.exists(save_path_dir):
            return

        current_checkpoint_filename = os.path.basename(current_checkpoint_full_path)
        deleted_count = 0
        for f_name in os.listdir(save_path_dir):
            if f_name.endswith('.pt') and f_name != current_checkpoint_filename:
                try:
                    file_to_delete = os.path.join(save_path_dir, f_name)
                    os.remove(file_to_delete)
                    # print(f"Rank {self.rank}: Deleted old checkpoint {file_to_delete}")
                    deleted_count += 1
                except OSError as del_e:
                    print(f"Rank {self.rank}: Warning - Failed to delete old checkpoint {file_to_delete}: {del_e}")
        if deleted_count > 0:
            print(f"Rank {self.rank}: Deleted {deleted_count} other checkpoint(s) in {save_path_dir}.")

    def _load_checkpoint(self):
        return ## TODO
        save_path_dir = os.path.join(self.config.save_dir,
                                 self.config.wandb_project if self.config.wandb_project else 'unnamed',
                                 self.config.wandb_name if self.config.wandb_name else 'unnamed',
                                 str(self.rank))

        if not os.path.exists(save_path_dir):
            print(f"Rank {self.rank}: Checkpoint directory {save_path_dir} not found. Starting from scratch.")
            return False

        checkpoint_files = []
        for f_name in os.listdir(save_path_dir):
            if f_name.endswith('.pt'):
                try:
                    step_num = int(f_name.split('.')[0])
                    checkpoint_files.append((step_num, f_name))
                except ValueError:
                    # Not a valid checkpoint file name pattern
                    continue
        
        # Sort by step number in descending order (latest first)
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)

        loaded_successfully = False
        for step_num, f_name in checkpoint_files:
            full_checkpoint_path = os.path.join(save_path_dir, f_name)
            try:
                print(f"Rank {self.rank}: Attempting to load checkpoint from {full_checkpoint_path}")
                checkpoint = torch.load(full_checkpoint_path, map_location=self.device)

                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.strategy.optim.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in checkpoint and self.strategy.scheduler is not None:
                    self.strategy.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                self.local_step = checkpoint['local_step']
                self.epoch = checkpoint['epoch']
                
                torch.set_rng_state(checkpoint['rng_state'].cpu()) # Ensure RNG state is on CPU before loading
                if self.device.type == 'cuda' and 'cuda_rng_state' in checkpoint:
                    if isinstance(checkpoint['cuda_rng_state'], torch.Tensor):
                        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu(), device=self.device) 
                    else:
                        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'], device=self.device)

                self.train_data_iter = iter(self.train_dataloader)
                self.val_data_iter = iter(self.val_dataloader)

                if len(self.train_dataloader) > 0:
                    batches_to_skip = self.local_step % len(self.train_dataloader)
                    print(f"Rank {self.rank}: Restored to epoch {self.epoch}, step {self.local_step}. Skipping {batches_to_skip} batches.")
                    for _ in range(batches_to_skip):
                        try:
                            next(self.train_data_iter)
                        except StopIteration:
                            print(f"Rank {self.rank}: Warning - StopIteration while fast-forwarding train_data_iter.")
                            break 
                else:
                    print(f"Rank {self.rank}: Restored to epoch {self.epoch}, step {self.local_step}. Train dataloader empty.")

                if self.rank == 0 and hasattr(self.logger, 'set_step'):
                    self.logger.set_step(self.local_step)
                elif self.rank == 0:
                    print(f"Rank 0: Logger step will resume from loaded local_step: {self.local_step}")

                print(f"Rank {self.rank}: Successfully loaded checkpoint {f_name}. Resuming at epoch {self.epoch}, step {self.local_step}.")
                loaded_successfully = True
                break # Exit loop once a checkpoint is successfully loaded
            except (RuntimeError, EOFError, zipfile.BadZipFile) as e: # Catch specific errors related to corrupted files
                print(f"Rank {self.rank}: Failed to load checkpoint {full_checkpoint_path}: {e}. Trying next available checkpoint.")
                # Optionally, delete the corrupted checkpoint file
                try:
                    os.remove(full_checkpoint_path)
                    print(f"Rank {self.rank}: Deleted corrupted checkpoint {full_checkpoint_path}.")
                except OSError as del_e:
                    print(f"Rank {self.rank}: Warning - Failed to delete corrupted checkpoint {full_checkpoint_path}: {del_e}")
            except Exception as e: # Catch any other unexpected error during loading
                print(f"Rank {self.rank}: An unexpected error occurred while loading checkpoint {full_checkpoint_path}: {e}. Trying next.")
                # Optionally, delete or move the problematic checkpoint

        if not loaded_successfully:
            print(f"Rank {self.rank}: No valid checkpoint found in {save_path_dir} after trying all options. Starting from scratch.")
            # Reset relevant states if starting from scratch, though __init__ defaults should cover this.
            self.local_step = 0 
            self.epoch = 0
            return False
        
        return True

    def _correlation_calculation(self):
        return ## TODO
        if self.num_nodes < 2:
            raise Exception('Correlation calculation cannot be used with < 2 nodes')
        
        # Ensure correlation is only calculated if interval is set
        if not self.config.correlation_interval:
             return None
        
        # Create a temporary directory for this timestep's checkpoints
        tmp_dir = os.path.join(self.config.save_dir, f"tmp_corr_{self.local_step}")
        # Only rank 0 creates the directory to avoid race conditions
        if self.rank == 0:
            os.makedirs(tmp_dir, exist_ok=True)
        torch.distributed.barrier() # Wait for rank 0 to create dir

        # Save model state dict for each rank
        checkpoint_path = os.path.join(tmp_dir, f"{self.rank}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)

        # Wait for all processes to save their checkpoints
        torch.distributed.barrier()

        corr_value = None
        if self.rank == 0:
            # Load all models as vectors
            model_vectors = []
            for r in range(self.config.num_nodes):
                model_path = os.path.join(tmp_dir, f"{r}.pt")
                # Ensure the file exists before trying to load
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location='cpu')
                    vector_list = []
                    for key in sorted(checkpoint.keys()):
                        value = checkpoint[key]
                        if isinstance(value, torch.Tensor):
                            vector_list.append(value.cpu().numpy().ravel())
                    if vector_list: # Check if we actually got any tensors
                        model_vectors.append(np.concatenate(vector_list))
                else:
                    print(f"Warning: Checkpoint file {model_path} not found for rank {r}.")


            if len(model_vectors) >= 2: # Need at least two models to compare
                # Calculate correlations between all pairs
                correlations = []
                for i in range(len(model_vectors)):
                    for j in range(i+1, len(model_vectors)):
                        corr = np.corrcoef(model_vectors[i], model_vectors[j])[0, 1]
                        correlations.append(corr)

                if correlations: # Ensure correlations list is not empty
                    corr_value = np.mean(correlations)

                    # Log average correlation to wandb using the logger
                    if self.logger:
                         self.logger.log(data={'avg_model_correlation': corr_value})
                else:
                    print("Warning: Could not calculate correlation, not enough valid model pairs.")
            else:
                 print(f"Warning: Not enough models loaded ({len(model_vectors)}) to calculate correlation.")


            # Clean up temporary directory
            import shutil
            shutil.rmtree(tmp_dir)

        # Wait for rank 0 to finish cleanup
        torch.distributed.barrier()

        return corr_value # Only rank 0 returns a value, others return None

    def train(self):
        if self.max_steps is None:
            self.max_steps = self.num_epochs * len(self.train_dataloader) / (self.batch_size // self.minibatch_size)

        self.strategy.max_steps = self.max_steps

        if self.rank == 0:
            if self.kwargs.get('wandb_project', None) is not None:
                self.logger = WandbLogger(model=self.model, 
                                    max_steps=self.max_steps,
                                    strategy=self.strategy,
                                    train_node=self,
                                    wandb_project=self.kwargs.get('wandb_project', None),
                                    wandb_name=self.kwargs.get('wandb_name', None))
            else:
                self.logger = Logger(model=self.model, 
                                    max_steps=self.max_steps)
                self.strategy.lr_callbacks.append(self.logger.log_lr)

        while self.local_step < self.max_steps:
            if self.local_step % self.val_interval == 0:
                self._evaluate()

            self._train_step()

            self.local_step += 1
            if self.rank == 0:
                self.logger.increment_step()

            # Calculate correlation if interval is set and it's time
            # if self.config.correlation_interval and self.local_step > 0 and self.local_step % self.config.correlation_interval == 0:
            #     self._correlation_calculation()

            dist.barrier()

            def print_dataset_size(dataset: torch.utils.data.Dataset):
                import pickle, sys, io

                buffer = io.BytesIO()
                pickle.dump(dataset, buffer, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Dataset size: {buffer.tell() // 1024 // 1024} MB")


        self._evaluate()

        # if self.config.checkpoint_interval is not None:
        #     self._save_checkpoint()


    def __config__(self):
        remove_keys = ['model', 'train_dataloader', 'val_dataloader', 'strategy']

        config = super().__config__(remove_keys=remove_keys)

        return config
