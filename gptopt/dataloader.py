# Based on https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py
# and https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
from pathlib import Path
import os
import torch
from torch.utils.data import IterableDataset
import torch.distributed as dist

magic_number = 20250401         # used in the header of saved binary files

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

        # get worker info for distributed training
        if dist.is_initialized():
            self.rank = dist.get_rank()           # Global rank of the process
            self.world_size = dist.get_world_size()  # Total number of processes
            self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
            print(f"Global Rank: {self.rank}, World Size: {self.world_size}, Local Rank: {self.local_rank}")
        else:
            print("Distributed package is not initialized.")
            self.rank = 0
            self.world_size = 1
        
        # get shards
        file_list = os.listdir(self.data_path)
        shard_list = sorted([s for s in file_list if split in s])
        self.shards = [os.path.join(self.data_path, s) for s in shard_list]
        self.n_shards = len(self.shards)
        self.reset()
        #print(f"Position initialized at {self.current_position} in rank {self.rank}")
  
        
    def reset(self):
        self.current_shard = 0
        self.current_position = self.B * self.T * self.rank
        self.tokens = load_data_shard(self.shards[self.current_shard], self.device)
        
    def next_batch(self):
        B, T = self.B, self.T
        rank, world_size = self.rank, self.world_size
        buf = self.tokens[self.current_position: self.current_position + B*T+1]
        x = (buf[:-1]).view(B, T).to(device=self.device, dtype=torch.int32, non_blocking=True) # inputs
        y = (buf[1:]).view(B, T).to(device=self.device, dtype=torch.int64, non_blocking=True) # targets
        self.current_position += B*T*self.world_size
        # move to next shard if next iteration will be out of bounds
        # TODO -- Q. does it not leave last few tokens in each file unprocessed
        if self.current_position + (B * T * self.world_size + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1)
            if self.current_shard >= self.n_shards:
                self.reset()    # end current epoch and reset for next epoch
                raise StopIteration
            self.tokens = load_data_shard(self.shards[self.current_shard], self.device)
            self.current_position = B * T * self.rank
        return x, y

    def get_state(self):
        return {'rank': self.rank,
                'position': self.current_position,
                'shard': self.current_shard
                }

    def __iter__(self):
        while True:
            try:
                batch = self.next_batch()
            except StopIteration:
                break  # End of epoch: exit the loop gracefully
            yield batch
        
