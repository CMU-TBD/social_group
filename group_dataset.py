import numpy as np
import torch
import torch.utils.data as data
from data_generation import DataGeneration as dg

class GroupDataset(data.Dataset):
    def __init__(self, set_ids):
        super(GroupDataset, self).__init__()
        history = 8
        offset = 2
        self.data_gen = dg(history, offset, set_ids, set_ids)

        self.set_ids = set_ids
        self._refresh_data()
        return

    def _refresh_data(self):
        self.count = 0
        self.all_inputs = []
        self.all_outputs = []
        for idx in self.set_ids:
            inputs, outputs = self.data_gen.generate_cases_all_groups(idx)
            self.all_inputs += inputs
            self.all_outputs += outputs
        self.data_len = len(self.all_inputs)
        return

    def __getitem__(self, idx):
        inputs = self.all_inputs[idx]
        outputs = self.all_outputs[idx]
        inputs = torch.from_numpy(np.transpose(inputs, (0, 3, 1, 2))).float()
        outputs = torch.from_numpy(np.transpose(outputs, (0, 3, 1, 2))).float()
        self.count += 1
        if self.count == self.data_len:
            self._refresh_data()
        return [idx, inputs, outputs]
    
    def __len__(self):
        return self.data_len
