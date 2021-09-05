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
    signals = torch.stack(signals)
    labels  = torch.stack(labels).long()

    return signals, labels




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
    weight_decay = 0.001
    num_epochs = 100
    PLOT_LOSS = False

    # Start within subject cross-validation scheme
    # For this eample, train with data from all positions from one subject.
    # Leave one repetition out for cross-validation
    between_subject_results = np.zeros((num_subjects, num_subjects))
    training_loss = np.zeros((num_subjects, num_epochs))
    validation_loss = np.zeros((num_subjects, num_epochs))

    for s_train in range(num_subjects):

        
        # We already have the dataset saved as .npy files from running main_construct_dataset.py
        # We can use a class to load in all the data, then use a data loader during training
        
        # The train rep are all reps that are not otherwise used.
        train_reps = list(range(1,num_reps+1))
        # The test rep is the rep of the loop r
        validation_rep = [train_reps.pop(-1)]

        validation_data = EMGData(s_train, chosen_rep_labels=validation_rep)
        training_data = EMGData(s_train, chosen_rep_labels=train_reps)
        
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

            training_loss[s_train,epoch] = train(model, training_loader, optimizer, device)
            validation_loss[s_train,epoch] = validate(model, validation_loader, device)

            scheduler.step(validation_loss[s_train,epoch])
        
        for s_test in range(num_subjects):
            testing_data = EMGData(s_test)
            test_loader  = build_data_loader(batch_size, num_workers, pin_memory, testing_data)
            between_subject_results[s_train,s_test] = test(model, test_loader, device)

    # I am planning on using the github readme file to keep track of the results of different pipelines, so let's output the results in markup format
    # Keep in mind, this table DOES currently include a "cheating" within-subject case. That entry should be completely omitted before outputting the table
    between_subject_results_1 = between_subject_results[~np.eye(between_subject_results.shape[0],dtype=bool)].reshape(between_subject_results.shape[0],-1)
    s_train_accuracy = np.mean(between_subject_results_1,axis=1) # average across testing subjects for each training subject
    between_subject_results_2 = between_subject_results[~np.eye(between_subject_results.shape[0],dtype=bool)].reshape(-1,between_subject_results.shape[1])
    s_test_accuracy  = np.mean(between_subject_results_2,axis=0) # average across training subjects for each test subject

    # Preface table with CNN
    print(f"## CNN")
    # Setup the header
    print("| train \ test | ", end='')
    for s in range(num_subjects):
        print(f" S{s} | ", end="")
    print(" Mean |")

    for s in range(num_subjects+2):
        print("| --- ", end='')
    print("|")

    for s_train in range(num_subjects):
        print(f"| S{s_train} | ",end="")
        for s_test in range(num_subjects):
            if s_train == s_test:
                print(f" NA | ",end="")
            else:
                print(f" {between_subject_results[s_train, s_test]} |", end="")

        print (f" {s_train_accuracy[s_train]} |")

    print("| Mean | ",end="")
    for s_test in range(num_subjects):
        print(f" {s_test_accuracy[s_test]} |", end="")

    
    
    np.save("Results/betweensubject_deeplearning.npy", between_subject_results)