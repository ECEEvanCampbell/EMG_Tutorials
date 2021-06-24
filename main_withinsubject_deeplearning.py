import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.batchnorm import BatchNorm1d

from torchsample.modules import ModuleTrainer

from dataset import EMGData
from utils import fix_random_seed
import numpy as np
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

#torch.backends.cudnn.enabled = False

class DeepLearningModel(nn.Module):
    def __init__(self, n_output, n_channels, n_input=64):
        super().__init__()
        # How many filters we want
        input_0 = n_channels
        input_1 = n_input
        input_2 = n_input//4
        input_3 = n_input//8

        # What layers do we want
        self.conv1 = nn.Conv1d(input_0, input_1, kernel_size=3)
        self.bn1   = nn.BatchNorm1d(input_1)

        self.conv2 = nn.Conv1d(input_1, input_2, kernel_size=3)
        self.bn2   = nn.BatchNorm1d(input_2)

        self.conv3 = nn.Conv1d(input_2, input_3, kernel_size=3)
        self.bn3   = nn.BatchNorm1d(input_3)

        # Get convolutional style output into linear format
        self.conv2linear = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(input_3, n_output)
        self.drop = nn.Dropout(p=0.2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.drop(x)

        x = self.conv2linear(x)
        x = x.permute(0,2,1)
       
        x = self.fc1(x)
        x = F.log_softmax(x, dim=2)

        return x



def build_data_loader(batch_size, num_workers, pin_memory, data):
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle = True,
        collate_fn = collate_fn,
        num_workers = num_workers,
        pin_memory = pin_memory
    )
    return data_loader

def collate_fn(batch):
    signals, labels = [], []
    # Populate these lists from the batch
    for signal, label, position in batch:
        # Concate signal onto list signals
        signals += [signal]
        labels  += [label]
   
    # Convert lists to tensors
    signals = pad_sequence(signals)
    labels  = torch.stack(labels).long()

    return signals, labels

def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0,2,1)


def train(model, training_loader, optimizer, device):
    model.train()
    losses = []

    for batch_idx, (data, label) in enumerate(training_loader):
        if data.shape[0]==1:
            continue

        data = data.to(device)
        label = label.to(device)

        output = model(data)
        # Output: (batch_size, 1, n_class)
        loss = F.nll_loss(output.squeeze(), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return sum(losses)/len(losses)


def validate(model, validation_loader, device):
    model.eval()
    losses = []

    for batch_idx, (data, label) in enumerate(validation_loader):
        if data.shape[0]==1:
            continue

        data = data.to(device)
        label = label.to(device)

        output = model(data)
        # Output: (batch_size, 1, n_class)
        loss = F.nll_loss(output.squeeze(), label)

        losses.append(loss.item())

    return sum(losses)/len(losses)



def test(model, test_loader, device):
    model.eval()
    correct = 0

    for batch_idx, (data, label) in enumerate(test_loader):
        if data.shape[0]==1:
            continue

        data = data.to(device)
        label = label.to(device)

        output = model(data)
        predictions = output.argmax(dim=-1)
        for i, prediction in enumerate(predictions):
            correct += int(prediction == label[i])



    return float(correct/ len(test_loader.dataset))


if __name__ == "__main__":
    # Fix the random seed -- make results reproducible
    # Found in utils.py, this sets the seed for the random, torch, and numpy libraries.
    fix_random_seed(1, torch.cuda.is_available())
   
    # get device available for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # My PC sucks, so I can't run things in parallel. If yours can, uncomment these lines
        #num_workers = 2
        #pin_memory  = True
        num_workers = 0
        pin_memory = False
    else:
        num_workers = 0
        pin_memory  = False

    # Dataset details, packaged together to easily pass them through functions if required.
    num_subjects  = 10
    num_channels  = 6
    num_motions   = 8
    motion_list   = ["wrist flexion","wrist extension","wrist supination","wrist pronation",
                     "power grip","pinch grip","hand open","no motion"] # This is the order as listed in the paper, check this
    num_reps      = 4
    num_positions = 16
    position_list = ["P1", "P2","P3","P4","P5","P6","P7","P8","P9","P10","P11","P12","P13","P14","P15","P16"]
    sampling_frequency = 1000
    winsize = 250
    wininc = 100
    dataset_characteristics = (num_subjects, num_motions, motion_list, num_reps, num_positions, position_list, winsize, wininc, sampling_frequency)

    # Deep learning variables:
    batch_size = 32
    lr = 0.005
    weight_decay = 0.001
    num_epochs = 100
    PLOT_LOSS = False

    # Start within subject cross-validation scheme
    # For this eample, train with data from all positions from one subject.
    # Leave one repetition out for cross-validation
    within_subject_results = np.zeros((num_subjects, num_reps))
    training_loss = np.zeros((num_subjects, num_reps, num_epochs))
    validation_loss = np.zeros((num_subjects, num_reps, num_epochs))

    for s in range(num_subjects):

        for r in range(num_reps):
            # We already have the dataset saved as .npy files from running main_construct_dataset.py
            # We can use a class to load in all the data, then use a data loader during training
           
            # The train rep are all reps that are not otherwise used.
            train_reps = list(range(1,num_reps+1))
            # The test rep is the rep of the loop r
            test_rep = [train_reps.pop(r)]
            # The validation rep is the final element of the train reps at this stage (removed by pop.)
            validation_rep = [train_reps.pop(-1)]

            testing_data = EMGData(s, chosen_rep_labels=test_rep)
            validation_data = EMGData(s, chosen_rep_labels=validation_rep)
            training_data = EMGData(s, chosen_rep_labels=train_reps)
           
            test_loader       = build_data_loader(batch_size, num_workers, pin_memory, testing_data)
            validation_loader = build_data_loader(batch_size, num_workers, pin_memory, validation_data)
            training_loader   = build_data_loader(batch_size, num_workers, pin_memory, training_data)
           
            # Define deep learning model structure and forward pass function
            model = DeepLearningModel(n_output=num_motions, n_channels=num_channels)
            # send to gpu
            model.to(device)

            # training setup
            optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=0.02, patience=3, factor=0.2)

            # Training Loop:
            for epoch in range(0, num_epochs):

                training_loss[s,r,epoch] = train(model, training_loader, optimizer, device)
                validation_loss[s,r,epoch] = validate(model, validation_loader, device)

                scheduler.step(validation_loss[s,r,epoch])
            
            if PLOT_LOSS:
                train_line      = plt.plot(training_loss[s,r,:],label="Train Loss")
                validation_line = plt.plot(validation_loss[s,r,:],label="Validation Loss")
                plt.legend()
                plt.show()
            within_subject_results[s,r] = test(model, test_loader, device)

    subject_accuracy = np.mean(within_subject_results, axis=1) * 100

    print("| Subject | ", end='')
    print(" CNN |")
    for c in range(1+1):
        print("| --- ",end='')
    print("|")
    for s in range(num_subjects):
        print("| S{} |".format(str(s)), end='')
        print(" {} |".format(str(subject_accuracy[s])))
    print("| Mean | ",end="")
    print(" {} |".format(str(np.mean(subject_accuracy))))
    print("| STD | ",end="")
    print(" {} |".format(str(np.std(subject_accuracy))))

    np.save("Results/withinsubject_deeplearning.npy", within_subject_results)