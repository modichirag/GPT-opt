# Based on https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py
# and https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
from pathlib import Path
import os
import torch
from torch.utils.data import IterableDataset
import torch.distributed as dist
from .utils import get_worker_info

magic_number = 20250401         # used in the header of saved binary files
DATA_DIR = "/mnt/ceph/users/cmodi/huggingface/"


def load_data_shard(filename, device):
    header = torch.from_file(filename, False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == magic_number, f"magic number mismatch in the data .bin file, expected {magic_number}, recieved {header[0]}"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with open(filename, "rb", buffering=0) as f:
        if ('gpu' in device) or ('cuda' in device):     # avoid pin_memory copy on gpu
            tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        else:
            tokens = torch.empty(num_tokens, dtype=torch.uint16)
        f.seek(256 * 4)                     # skip over header
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

    
class ShardedDataLoader(IterableDataset):

    def __init__(self, data_path, B, T, split, device):
        self.data_path = data_path
        self.B = B
        self.T = T
        self.split = split
        assert split in ('train', 'val')
        self.device = device
        self.world_size, self.rank, self.local_rank, self.device = get_worker_info()
        
        # get shards
        file_list = os.listdir(self.data_path)
        shard_list = sorted([s for s in file_list if split in s])
        self.shards = [os.path.join(self.data_path, s) for s in shard_list]
        self.n_shards = len(self.shards)
        self.get_length()
        self.reset()
        print(f"Initialized {split} dataloader in {self.rank} at : ", self.get_state())
        
    def reset(self):
        self.current_shard = 0
        self.current_position = self.B * self.T * self.rank
        self.last_token_position = self.B * self.T * self.world_size
        self.tokens = load_data_shard(self.shards[self.current_shard], self.device)

    def get_state(self):
        return {'rank': self.rank,
                'position': self.current_position,
                'shard': self.current_shard
                }

    def set_state(self, state):
        assert self.rank == state['rank']
        self.current_position = state['position']
        self.current_shard = state['shard']
            
    def get_length(self):
        tokens = load_data_shard(self.shards[0], self.device)
        base_length = len(tokens)
        if self.n_shards == 1:
            self.size = int(base_length)
        else:
            # assumes all shards except last are of same length
            tokens = load_data_shard(self.shards[-1], self.device)
            self.size = int(base_length*(self.n_shards - 1) + len(tokens))

    def __len__(self):
        return self.size
        
        
    def next_batch(self):

        B, T = self.B, self.T
        # Check if we can sample entire global batch from this shard
        if self.last_token_position >= len(self.tokens):
            if self.current_shard == self.n_shards - 1:
                self.reset()    # end current epoch and reset for next epoch
                print(f"Rank {self.rank} reached end of {self.split} dataloader. Resetting to : ", self.get_state())
                raise StopIteration
            else:
                self.current_shard = (self.current_shard + 1)
                self.tokens = load_data_shard(self.shards[self.current_shard], self.device)
                self.current_position = B * T * self.rank
                self.last_token_position =  B * T * self.world_size
                
        buf = self.tokens[self.current_position: self.current_position + B*T+1]
        x = (buf[:-1]).view(B, T).to(device=self.device, dtype=torch.int64, non_blocking=True) # inputs
        y = (buf[1:]).view(B, T).to(device=self.device, dtype=torch.int64, non_blocking=True) # targets
        self.current_position += B * T * self.world_size
        self.last_token_position += B * T * self.world_size
        return x, y

    def __iter__(self):
        while True:
            try:
                batch = self.next_batch()
            except StopIteration:
                break  # End of epoch: exit the loop gracefully
            yield batch
        
    # def __next__(self):
    #     try:
    #         batch = self.next_batch()
    #     except StopIteration:
    #         break  # End of epoch: exit the loop gracefully
    #     return batch
        
