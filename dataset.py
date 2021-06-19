import torch
import numpy as np
from torch.utils.data import Dataset


class EMGData(Dataset):
    def __init__(self, subject_number, chosen_rep_labels=None, chosen_pos_labels=None):

        if isinstance(subject_number, list):
            data = []
            for n in subject_number:
                data.append(np.load("Data/S{}.npy".format(str(n)), allow_pickle=True))

            subject_data = np.concatenate(data)

        else:
            subject_data = np.load("Data/S{}.npy".format(str(subject_number)), allow_pickle=True)

        if chosen_rep_labels is not None:
            subject_data = [i for i in subject_data if i[2] in chosen_rep_labels]

        if chosen_pos_labels is not None:
            subject_data = [i for i in subject_data if i[5] in chosen_pos_labels]
            
        self.subject_number = subject_number

        # extract classes
        self.class_label = torch.tensor([i[1] for i in subject_data], dtype=torch.float)
        self.rep_label   = torch.tensor(i[2] for i in subject_data], dtype=torch.int)
        self.num_labels = torch.unique(self.class_label).shape[0]

        self.position_label = torch.tensor([i[5] for i in subject_data])
        self.num_position = torch.unique(self.intensity_label).shape[0]
        self.data = torch.tensor([i[4] for i in subject_data], dtype=torch.float)

        # set signal info vars
        self.sig_length = self.data.shape[-1]
        self.num_channels = self.data.shape[-2]

    def __len__(self):
        return len(self.class_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx]
        labels = self.class_label[idx]
        position = self.position_label[idx]

        return data, labels, position