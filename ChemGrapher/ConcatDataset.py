from torch.utils.data import Dataset

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lengths = []
        for index,d in enumerate(self.datasets):
            self.lengths.append(len(d))

    def __getitem__(self, i):
        tot_len = 0
        found_index = 0
        for index,l in enumerate(self.lengths):
            tot_len = tot_len + l
            if i < tot_len:
                found_index = index
                break
        if found_index == 0:
            return self.datasets[found_index][i]
        else:
            new_index = i - self.lengths[found_index-1]
            return self.datasets[found_index][new_index]

    def __len__(self):
        return sum([len(d) for d in self.datasets])
