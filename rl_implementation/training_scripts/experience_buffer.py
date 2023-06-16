import random
import torch

class ReplayMemory:
    
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        # print("printing batch inside sample() function: ", batch)
        return_ls = []
        for items in batch:
            # print("printing items: ", items)
            # print("printing concatenation: ", torch.cat(items))
            return_ls.append(torch.cat(items))
        # print("printing given concat operation: ", [torch.cat(items) for items in batch])
        # return [torch.cat(items) for items in batch]
        return return_ls

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        return len(self.memory)