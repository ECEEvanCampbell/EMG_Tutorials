import torch
from dataset import EMGData
from utils import fix_random_seed
import numpy as np





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

    # Deep learning variables:
    

    # Start within subject cross-validation scheme
    # For this eample, train with data from all positions from one subject.
    # Leave one repetition out for cross-validation
    within_subject_results = np.zeros((num_subjects, num_reps))

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
            
