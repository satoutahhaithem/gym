import torch

class CommunicationHandler:
    def __init__(self, config):
        self.config = config

        self.buffer = [{} for _ in range(self.config.num_nodes)]

    def post_tensor(self, name, tensor, rank):
        self.buffer[rank][name] = tensor

    def all_reduce_tensor(self, name):
        tensor_result = torch.zeros_like(self.buffer[0][name])
        n = 0
        
        for rank in range(self.config.num_nodes):
            if name in self.buffer[rank]:
                tensor_result += self.buffer[rank][name]
                n += 1

        return tensor_result, n