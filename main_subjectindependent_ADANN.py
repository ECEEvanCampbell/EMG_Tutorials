import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.batchnorm import BatchNorm1d

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
        # Forward pass: input x, output probabilities of predicted class
        # First layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.drop(x)

        # Second Layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.drop(x)

        # Third Layer
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.drop(x)

        # Convert to linear layer suitable input
        x = self.conv2linear(x)
        x = x.permute(0,2,1)
       
        # final layer: linear layer that outputs N_Class neurons
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
    # Train the model
    # model.train - enable gradient tracking, enable batch normalization, dropout
    model.train()
    # Store losses of this epoch in a list (element = loss on batch)
    losses = []

    for batch_idx, (data, label) in enumerate(training_loader):
        # Send data, labels to GPU if GPU is available
        data = data.to(device)
        label = label.to(device)
        # Passing data to model calls the forward method.
        output = model(data)
        # Output: (batch_size, 1, n_class)
        # Use negative log likelihood loss for training
        loss = F.nll_loss(output.squeeze(), label)
        # reset optimizer buffer
        optimizer.zero_grad()
        # Send the loss to the optimizer (direction of update for each neuron)
        loss.backward()
        # Update weights 
        optimizer.step()
        # Store the loss of this batch
        losses.append(loss.item())
    # Return the average training loss on this epoch
    return sum(losses)/len(losses)


def validate(model, validation_loader, device):
    # Evaluate the model
    # model.eval - disable gradient tracking, batch normalization, dropout
    model.eval()
    # Store losses of this epoch in a list (element = loss on batch)
    losses = []

    for batch_idx, (data, label) in enumerate(validation_loader):
        # Send data, labels to GPU if GPU is available
        data = data.to(device)
        label = label.to(device)
        # Passing data to model calls the forward method.
        output = model(data)
        # Output: (batch_size, 1, n_class)
        # Use negative log likelihood loss for validation
        loss = F.nll_loss(output.squeeze(), label)
        # Store the loss of this batch
        losses.append(loss.item())

    # Return the average validation loss on this epoch
    return sum(losses)/len(losses)



def test(model, test_loader, device):
    # Evaluate the model
    # model.eval - disable gradient tracking, batch normalization, dropout
    model.eval()
    # Keep track of correct samples
    correct = 0

    for batch_idx, (data, label) in enumerate(test_loader):
        # Send data, labels to GPU if GPU is available
        data = data.to(device)
        label = label.to(device)
        # Passing data to model calls the forward method.
        output = model(data)
        predictions = output.argmax(dim=-1)
        # Add up correct samples from batch
        for i, prediction in enumerate(predictions):
            correct += int(prediction == label[i])
    # Return average accuracy 
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
    lambda_val = 0.1
    weight_decay = 0.001
    num_epochs = 100
    PLOT_LOSS = False

    # Start within subject cross-validation scheme
    # For this eample, train with data from all positions from one subject.
    # Leave one repetition out for cross-validation
    independent_subject_results = np.zeros((num_subjects))
    # Unlike a standard CNN, we have a loss from the class prediction and a loss from the subject prediction
    training_class_loss = np.zeros((num_epochs))
    training_subject_loss = np.zeros((num_epochs))
    validation__class_loss = np.zeros((num_epochs))
    validation__subject_loss = np.zeros((num_epochs))
    # The pairs of subjects selected during training is random. Let's keep track of these in a list
    training_subjects_selected = []
    

    # An ADANN model will be trained dusing the data from all subjects (reps [1,2,3])
    # The model will be tested on rep 4 of all subjects (with the correct batch normalization parameters loaded for each subject)

    # The ADANN training process is like a regular CNN training process, but there are 2 special additions

    # I: Adversarial training:
    # Every epoch, the training data from two subjects will be randomly selected. In addition to the normal classification head,
    # There is also a domain head that is trying to predict the subject that the data came from.  The subject predictions are compared
    # against the true subject labels to get a subject loss expression.  The subject loss is inverted before being used to adapt the 
    # weights of the convolutional layers.  This inversion is what penalizes learned features from encoding subject-specific characteristics.
    # The important areas of the code for adversarial training to work are:
    # --

    # II: Adaptive Batch Normalization
    # During the training process, the model will have the data from different subjects being used as input at any given time.  When the data
    # from a subject is selected to be used in a particular epoch, the batch normalization parameters associated with that subject are loaded
    # into the model. The motivation for sharing model weights across subjects and changing batch normalization parameters between subjects 
    # is to encode gesture-specific information in the weights and subject-specific information in batch normalization parameters.
    # The important areas of code for adaptive batch normalization to work are:
    # --

    # 



    # Train process:

    # Test process:




    np.save("Results/subjectindependent_ADANN.npy", independent_subject_results)