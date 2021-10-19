import torch
import torch.nn as nn
from torch.nn import parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.batchnorm import BatchNorm1d

from dataset import EMGData
from utils import fix_random_seed
import numpy as np
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from torch.autograd import Function

import copy

# Define a layer that records the negative gradient for training subject encodings
class ReversalGradientLayerF(Function):
    @staticmethod
    def forward(ctx, input, lambda_hyper_parameter):
        ctx.lambda_hyper_parameter = lambda_hyper_parameter
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_hyper_parameter
        return output, None

    @staticmethod
    def grad_reverse(x, constant):
        # Constant is lambda value from paper.
        return ReversalGradientLayerF.apply(x, constant)


class ADANNModel(nn.Module):
    def __init__(self, n_output, n_channels, input_dims, n_input=64, lambda_value=0.1, num_subjects=10, lr=1e-3):
        super().__init__()

        self.lambda_value = lambda_value
        self.num_subjects = num_subjects

        self.init_network(n_channels, n_input, n_output, input_dims)

        self.BN_dict = self.init_BN_dict()

        self.class_loss_fn = nn.CrossEntropyLoss()
        self.domain_loss_fn = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', threshold=0.02, patience=3, factor=0.2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def init_network(self, n_channels, n_input, n_output, input_dims):
        # How many filters we want
        input_0 = n_channels
        input_1 = n_input
        input_2 = n_input//4
        input_3 = n_input//8

        # What layers do we want
        self.conv1 = nn.Conv1d(input_0, input_1, kernel_size=3)
        self.bn1   = nn.BatchNorm1d(input_1, track_running_stats=False)

        self.conv2 = nn.Conv1d(input_1, input_2, kernel_size=3)
        self.bn2   = nn.BatchNorm1d(input_2, track_running_stats=False)

        self.conv3 = nn.Conv1d(input_2, input_3, kernel_size=3)
        self.bn3   = nn.BatchNorm1d(input_3, track_running_stats=False)

        # Get convolutional style output into linear format
        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        # Classification Head
        self.fc1 = nn.Linear(fc_input_dims, n_output)
        # Subject Head
        self.fc2 = nn.Linear(fc_input_dims, 2)

        self.drop = nn.Dropout(p=0.2)
        self.activation = nn.ReLU()

    def init_BN_dict(self):
        list_BN_dictionary = []
        BN_dictionary = {}
        for name, param in self.named_parameters():
            if 'bn' in name:
                # We collect
                BN_dictionary[name] = copy.deepcopy(param.data[:])
        # repeat dictionary for num_subjects
        for s in range(self.num_subjects):
            list_BN_dictionary.append(copy.deepcopy(BN_dictionary))

        return list_BN_dictionary

    def switch_BN_dict(self, subject_id):
        current_state = self.state_dict()
        for name in self.BN_dict[subject_id]:
            if name not in current_state:
                print('Error in BN state_dict format')
                raise(KeyError)
            params = copy.deepcopy(self.BN_dict[subject_id][name])
            self.state_dict()[name][:] = params
    
    def update_BN_dict(self, subject_id):
        current_state = self.state_dict()
        for name in self.BN_dict[subject_id]:
            self.BN_dict[subject_id][name] = copy.deepcopy(current_state[name])

    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

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

        # Flatten in preparation for linear layer
        x_f = x.view(x.size()[0], -1)
       
        # final layer: linear layer that outputs N_Class neurons
        y = self.fc1(x_f)

        # Subject Head - This is where the gradient is reversed to penalize subject specific information
        reversed_layer = ReversalGradientLayerF.grad_reverse(x_f, self.lambda_value)
        s = self.fc2(reversed_layer)

        # Return both class and subject
        return y,s



def build_data_loader(batch_size, num_workers=0, pin_memory=False, data=None):
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

# TODO: change from vanilla DL train to ADANN train
# Class loss & subject loss
# change BN parameters between subjects
def train(model, source_train_loader, adversarial_train_loader, subjects):
    # Train the model
    # model.train - enable gradient tracking, enable batch normalization, dropout
    model.train()
    # Store losses and accuracies of interest of this epoch in a list (element = loss on batch)
    loss_source_class = []
    loss_source_subject = []
    loss_adversarial_subject = []
    source_class_accuracy   = []
    source_subject_accuracy = []
    adversarial_subject_accuracy = []

    # We are now using 2 dataloaders, so we need to advance one in tandem of the other
    adversarial_iterable = iter(adversarial_train_loader)
    for batch_idx, (source_data, source_class) in enumerate(source_train_loader):

        model.switch_BN_dict(subjects[0])
        # First, deal with the batch of source training data 
        # Send data, labels to GPU if GPU is available
        source_data = source_data.to(model.device)
        source_class = source_class.to(model.device)
        source_subject = torch.zeros((source_data.shape[0]), dtype=torch.long).to(model.device)

        # Passing data to model calls the forward method.
        # Forward returns class logits and subject logits
        predicted_source_class, predicted_source_subject = model(source_data)
        
        # Compute loss for both class prediction and subject prediction of source
        source_loss_class   = model.class_loss_fn(predicted_source_class, source_class)
        source_loss_subject = 0.05 * model.domain_loss_fn(predicted_source_subject, source_subject)

        model.optimizer.zero_grad()
        
        # We can compute the backwards update for the classification loss now,
        # Retain graph is needed here, as source_loss_subject is not used yet so we can't clear buffers.
        source_loss_class.backward(retain_graph=True)
        source_loss_subject.backward()
        model.optimizer.step()

        

        # Now we save the BN change from the update
        model.update_BN_dict(subjects[0])

        # Now we move onto the adversarial subject
        # adversarial subject -- keep in mind, we only care about the domain loss for the adversarial subject.
        # We check accuracy here purely for curiosity, adversarial class labels are not used for weight/BN update
        model.switch_BN_dict(subjects[1])

        # This should be similar, just set up the required data
        adversarial_data, _ = next(adversarial_iterable)
        adversarial_data = adversarial_data.to(model.device)
        # The subject labels here are ones to provide a different label than the source subject
        adversarial_subject = torch.ones((adversarial_data.shape[0]), dtype=torch.long).to(model.device)
        # Feed the adversarial data into the model to get subject predictions
        _, predicted_adversarial_subject = model(adversarial_data)
        # Get domain loss
        adversarial_loss_subject = 0.05 * model.domain_loss_fn(predicted_adversarial_subject, adversarial_subject)


        adversarial_loss_subject.backward()
        model.optimizer.step()

        loss_source_class += [source_loss_class.item()]
        loss_source_subject += [source_loss_subject.item()]
        loss_adversarial_subject += [adversarial_loss_subject.item()]
        source_class_accuracy   += [( ((torch.argmax(predicted_source_class,dim=1)==source_class).sum())/predicted_source_class.shape[0] ).item()]
        source_subject_accuracy += [( ((torch.argmax(predicted_source_subject,dim=1)==source_subject).sum())/predicted_source_subject.shape[0] ).item()]
        adversarial_subject_accuracy += [( ((torch.argmax(predicted_adversarial_subject,dim=1)==adversarial_subject).sum())/predicted_adversarial_subject.shape[0] ).item()]
        

    # Return the average training loss on this epoch
    return np.array(loss_source_class).mean(), np.array(loss_source_subject).mean(), np.array(loss_adversarial_subject).mean(), \
        np.array(source_class_accuracy).mean(), np.array(source_subject_accuracy).mean(), np.array(adversarial_subject_accuracy).mean() 

# TODO: change from vanilla DL validate to ADANN validate
def validate(model, source_valid_loader, adversarial_valid_loader, device):
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

    train_rep = [1,2]
    val_rep   = [3]
    test_rep  = [4]

    # Deep learning variables:
    batch_size = 32
    lr = 0.005
    lambda_val = 0.1
    weight_decay = 0.001
    num_epochs = 1000
    PLOT_LOSS = False

    # Start within subject cross-validation scheme
    # Unlike a standard CNN, we have a loss from the class prediction and a loss from the subject prediction
    independent_subject_results = np.zeros((num_subjects))
    source_training_class_loss = np.zeros((num_epochs))
    source_training_subject_loss = np.zeros((num_epochs))
    source_validation_class_loss = np.zeros((num_epochs))
    source_validation_subject_loss = np.zeros((num_epochs))
    adversarial_training_class_loss = np.zeros((num_epochs))
    adversarial_training_subject_loss = np.zeros((num_epochs))
    adversarial_validation_class_loss = np.zeros((num_epochs))
    adversarial_validation_subject_loss = np.zeros((num_epochs))

    source_training_class_accuracy = np.zeros((num_epochs))
    source_training_subject_accuracy = np.zeros((num_epochs))
    source_validation_class_accuracy = np.zeros((num_epochs))
    source_validation_subject_accuracy = np.zeros((num_epochs))
    adversarial_training_subject_accuracy = np.zeros((num_epochs))
    adversarial_validation_subject_accuracy = np.zeros((num_epochs))
    
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

    model = ADANNModel(n_output=num_motions, n_channels=num_channels, num_subjects=num_subjects, input_dims=[num_channels, winsize], lr = lr)

    # training setup
    
    
    
    # Train process:
    for epoch in range(num_epochs):
        # First, select two subjects
        epoch_subjects = np.random.choice(np.array(range(num_subjects)), 2, replace=False)
        # Get the data for subject 0, and subject 1
        # These numbers later indicate what label the domain linear layer tries to predict
        source_subject = epoch_subjects[0]
        adversarial_subject = epoch_subjects[1]
        # This loading procedure adds a LARGE amount of time to the training procedure (repeated each epoch)
        # __ future work could be to find a method to improve this area while not requiring incredible amounts of memory.
        # These datasets only contain the training data ~ 50% of this dataset
        source_data_train         = EMGData(source_subject, chosen_rep_labels=train_rep)
        source_train_loader       = build_data_loader(batch_size,  data=source_data_train)
        adversarial_data_train   = EMGData(adversarial_subject, chosen_rep_labels=train_rep)
        adversarial_train_loader = build_data_loader(batch_size, data=adversarial_data_train)
        # These datasets only contain the validation data ~ 25% of this dataset
        source_data_valid         = EMGData(source_subject, chosen_rep_labels=val_rep)
        source_valid_loader       = build_data_loader(batch_size, data=source_data_valid)
        adversarial_data_valid   = EMGData(adversarial_subject, chosen_rep_labels=val_rep)
        adversarial_valid_loader = build_data_loader(batch_size, data=adversarial_data_valid)

        source_training_class_loss[epoch], source_training_subject_loss[epoch], adversarial_training_subject_loss[epoch], \
            source_training_class_accuracy[epoch], source_training_subject_accuracy[epoch], adversarial_training_subject_accuracy[epoch] = train(model, source_train_loader, adversarial_train_loader, epoch_subjects)
        source_validation_class_loss[epoch], source_validation_subject_loss[epoch], \
            adversarial_validation_class_loss[epoch], adversarial_validation_subject_loss[epoch] = validate(model, source_valid_loader, adversarial_valid_loader)
        
        validation_loss = (1-model.lambda_value) * (source_validation_class_loss[epoch])+ \
            (model.lambda_value) * (source_validation_subject_loss[epoch] + adversarial_validation_subject_loss[epoch])/2

        model.scheduler.step(validation_loss)
        # Add a nice print statement

    # Test process:




    np.save("Results/subjectindependent_ADANN.npy", independent_subject_results)