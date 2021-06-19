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

    # Handcrafted feature variables:
    featuresets = ["TD","TDPSD","LSF4","LSF9"]
    num_featuresets = len(featuresets)

    # Start within subject cross-validation scheme
    # For this eample, train with data from all positions from one subject.
    # Leave one repetition out for cross-validation
    within_subject_results = np.zeros((num_subjects, num_reps, num_featuresets))

    for s in range(num_reps):

        for r in range(num_reps):
            train_reps = list(range(1, num_reps+1))
            test_reps = [train_reps.pop(r)]
            training_data = EMGData(s, chosen_rep_labels=train_reps)
            testing_data  = EMGData(s, chosen_rep_labels=test_reps)