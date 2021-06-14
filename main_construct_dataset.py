import os
import numpy as np
import math
import torch
from utils import fix_random_seed

def load_subject(subject_id, intensity_list,winsize,wininc,base_dir = "Data/Raw_Data"):

    # Inside of dataset folder, get list of all files associated with the subject of subject_id
    subj_path = os.listdir(base_dir+'/S' + str(subject_id + 1))
    training_data = []
    # For this list:
    for f in subj_path:
        # Get the identifiers in the filename
        path = os.path.join(base_dir,"S"+ str(subject_id+1),f)
        class_num = int(f.split('_')[1][1:])
        rep_num   = int(f.split('_')[3][1])
        position = int(f.split('_')[2][1:])

        # If the file meets the inclusion criteria 
        

        # load the file
        data = np.genfromtxt(path,delimiter=',')
        num_windows = math.floor((data.shape[0]-winsize)/wininc)

        st=0
        ed=st+winsize
        for w in range(num_windows):
            training_data.append([subject_id,class_num-1, rep_num,w,data[st:ed,:].transpose(), position])
            st = st+wininc
            ed = ed+wininc

    np.random.shuffle(training_data)
    np.save("Data/S"+str(subject_id), training_data)
    return training_data


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

    for s in range(num_subjects):
        if os.path.exists("Data/S{}.npy".format(str(s))):
            print("Subject {} is already prepared".format(str(s)))
        else:
            load_subject(s, position_list,winsize,wininc,base_dir = "Data/Raw_Data")
            print("Subject {} was prepared".format(str(s)))


    print('Dataset is ready for EMG gesture recognition pipelines')