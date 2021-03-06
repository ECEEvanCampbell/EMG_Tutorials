import torch
from dataset import EMGData
from utils import fix_random_seed
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import time


def extract_TD_feats(signals, num_channels):

    features = np.zeros((signals.shape[0],num_channels*4),dtype =float)
    if torch.is_tensor(signals):
        signals = signals.numpy()
    features[:,0:num_channels]  = getMAVfeat(signals)
    features[:,num_channels:2*num_channels] = getZCfeat(signals)
    features[:,2*num_channels:3*num_channels] = getSSCfeat(signals)
    features[:,3*num_channels:4*num_channels] = getWLfeat(signals)

    return features

def extract_TDPSD_feats(signals, num_channels):
    # There are 6 features per channel
    features = np.zeros((signals.shape[0], num_channels*6), dtype=float)
    if torch.is_tensor(signals):
        signals = signals.numpy()
    # TDPSD feature set adapted from: https://github.com/RamiKhushaba/getTDPSDfeat
    # Extract the features from original signal and nonlinear version
    ebp = KSM1(signals)
    # np.spacing = epsilon (smallest value), done so log does not return inf.
    efp = KSM1(np.log(signals**2 + np.spacing(1)))
    # Correlation analysis:
    num = -2*np.multiply(efp, ebp)
    den = np.multiply(efp, efp) + np.multiply(ebp,ebp)
    #Feature extraction goes here
    features = num-den
    return features

def KSM1(signals):
    samples = signals.shape[2]
    channels = signals.shape[1]
    # Root squared zero moment normalized
    m0 = np.sqrt(np.sum(signals**2,axis=2))
    m0 = m0 ** 0.1 / 0.1
    # Prepare derivatives for higher order moments
    d1 = np.diff(signals, n=1, axis=2)
    d2 = np.diff(d1     , n=1, axis=2)
    # Root squared 2nd and 4th order moments normalized
    m2 = np.sqrt(np.sum(d1 **2, axis=2)/ (samples-1))
    m2 = m2 ** 0.1 / 0.1
    m4 = np.sqrt(np.sum(d2**2,axis=2) / (samples-1))
    m4 = m4 **0.1/0.1

    # Sparseness
    sparsi = m0/np.sqrt(np.abs((m0-m2)*(m0-m4)))

    # Irregularity factor
    IRF = m2/np.sqrt(np.multiply(m0,m4))

    # Waveform Length Ratio
    WLR = np.sum( np.abs(d1),axis=2)-np.sum(np.abs(d2),axis=2)

    Feat = np.concatenate((m0, m0-m2, m0-m4, sparsi, IRF, WLR), axis=1)
    Feat = np.log(np.abs(Feat))
    return Feat



def extract_LSF4_feats(signals, num_channels):

    features = np.zeros((signals.shape[0],num_channels*4),dtype =float)
    if torch.is_tensor(signals):
        signals = signals.numpy()
    features[:,0:num_channels]  = getLSfeat(signals)
    features[:,num_channels:2*num_channels] = getMFLfeat(signals)
    features[:,2*num_channels:3*num_channels] = getMSRfeat(signals)
    features[:,3*num_channels:4*num_channels] = getWAMPfeat(signals)

    return features

def extract_LSF9_feats(signals, num_channels):

    features = np.zeros((signals.shape[0],num_channels*9),dtype =float)
    if torch.is_tensor(signals):
        signals = signals.numpy()
    features[:,0:num_channels]   = getLSfeat(signals)
    features[:,num_channels:2*num_channels]  = getMFLfeat(signals)
    features[:,2*num_channels:3*num_channels] = getMSRfeat(signals)
    features[:,3*num_channels:4*num_channels] = getWAMPfeat(signals)
    features[:,4*num_channels:5*num_channels] = getZCfeat(signals)
    features[:,5*num_channels:6*num_channels] = getRMSfeat(signals)
    features[:,6*num_channels:7*num_channels] = getIAVfeat(signals)
    features[:,7*num_channels:8*num_channels] = getDASDVfeat(signals)
    features[:,8*num_channels:9*num_channels] = getVARfeat(signals)

    return features

def getMAVfeat(signal):
    feat = np.mean(np.abs(signal),2)
    return feat


def getZCfeat(signal):
    sgn_change = np.diff(np.sign(signal),axis=2)
    neg_change = sgn_change == -2
    pos_change = sgn_change ==  2
    feat_a = np.sum(neg_change,2)
    feat_b = np.sum(pos_change,2)
    return feat_a+feat_b


def getSSCfeat(signal):
    d_sig = np.diff(signal,axis=2)
    return getZCfeat(d_sig)


def getWLfeat(signal):
    feat = np.sum(np.abs(np.diff(signal,axis=2)),2)
    return feat




def getLSfeat(signal):
    feat = np.zeros((signal.shape[0],signal.shape[1]))
    for w in range(0, signal.shape[0],1):
        for c in range(0, signal.shape[1],1):
            tmp = lmom(np.reshape(signal[w,c,:],(1,signal.shape[2])),2)
            feat[w,c] = tmp[0,1]
    return feat

def lmom(signal, nL):
    # same output to matlab when ones vector of various sizes are input
    b = np.zeros((1,nL-1))
    l = np.zeros((1,nL-1))
    b0 = np.zeros((1,1))
    b0[0,0] = np.mean(signal)
    n = signal.shape[1]
    signal = np.sort(signal, axis=1)
    for r in range(1,nL,1):
        num = np.tile(np.asarray(range(r+1,n+1)),(r,1))  - np.tile(np.asarray(range(1,r+1)),(1,n-r))
        num = np.prod(num,axis=0)
        den = np.tile(np.asarray(n),(1,r)) - np.asarray(range(1,r+1))
        den = np.prod(den)
        b[r-1] = 1/n * np.sum(num / den * signal[0,r:n])
    tB = np.concatenate((b0,b))
    B = np.flip(tB,0)
    for i in range(1, nL, 1):
        Spc = np.zeros((B.shape[0]-(i+1),1))
        Coeff = np.concatenate((Spc, LegendreShiftPoly(i)))
        l[0,i-1] = np.sum(Coeff * B)
    L = np.concatenate((b0, l),1)

    return L

def LegendreShiftPoly(n):
    # Verified: this has identical function to MATLAB function for n = 2:10 (only 2 is used to compute LS feature)
    pk = np.zeros((n+1,1))
    if n == 0:
        pk = 1
    elif n == 1:
        pk[0,0] = 2
        pk[1,0] = -1
    else:
        pkm2 = np.zeros(n+1)
        pkm2[n] = 1
        pkm1 = np.zeros(n+1)
        pkm1[n] = -1
        pkm1[n-1] = 2

        for k in range(2,n+1,1):
            pk = np.zeros((n+1,1))
            for e in range(n-k+1,n+1,1):
                pk[e-1] = (4*k-2)*pkm1[e]+ (1-2*k)*pkm1[e-1] + (1-k) * pkm2[e-1]
            pk[n,0] = (1-2*k)*pkm1[n] + (1-k)*pkm2[n]
            pk = pk/k

            if k < n:
                pkm2 = pkm1
                pkm1 = pk

    return pk

def getMFLfeat(signal):
    feat = np.log10(np.sum(np.abs(np.diff(signal, axis=2)),axis=2))
    return feat

def getMSRfeat(signal):
     feat = np.abs(np.mean(np.sqrt(signal.astype('complex')),axis=2))
     return feat

def getWAMPfeat(signal,threshold=2e-3): # TODO: add optimization if threshold not passed, need class labels
    feat = np.sum(np.abs(np.diff(signal, axis=2)) > threshold, axis=2)
    return feat

def getRMSfeat(signal):
    feat = np.sqrt(np.mean(np.square(signal),2))
    return feat

def getIAVfeat(signal):
    feat = np.sum(np.abs(signal),axis=2)
    return feat

def getDASDVfeat(signal):
    feat = np.sqrt(np.mean(np.diff(np.square(signal.astype('complex')),2),2))
    return feat

def getVARfeat(signal):
    feat = np.var(signal,axis=2)
    return feat







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
    featuresets = ["TD", "TDPSD","LSF4","LSF9"]
    num_featuresets = len(featuresets)
    featureset_times = []

    # Start within subject cross-validation scheme
    # For this eample, train with data from all positions from one subject.
    # Leave one repetition out for cross-validation
    within_subject_results = np.zeros((num_subjects, num_reps, num_featuresets))
    
    for s in range(num_subjects):
        subject_dataset = EMGData(s)
        subject_data = subject_dataset.data
        subject_class = subject_dataset.class_label 
        subject_rep   = subject_dataset.rep_label

        for f in range(num_featuresets):
            if s==0:
                # Timing starts
                t0 = time.perf_counter()

            if featuresets[f] == "TD":
                features = extract_TD_feats(subject_data, num_channels)
            elif featuresets[f] == "TDPSD": # TODO: TDPSD is a common feature set -- should be added to tutorial.
                features = extract_TDPSD_feats(subject_data, num_channels)
            elif featuresets[f] == "LSF4":
                features = extract_LSF4_feats(subject_data, num_channels)
            elif featuresets[f] == "LSF9":
                features = extract_LSF9_feats(subject_data, num_channels)
            else:
                print("Unknown featureset given: {}".format(featuresets[f]))
                continue
            
            if s == 0:
                featureset_times.append((time.perf_counter()-t0)/subject_data.shape[0])

            # Now that we have the features extracted, we can partition the dataset into a training and testing set using
            # leave-one-repetition-out cross-validation.

            for r in range(num_reps):
                # The training reps are all reps except rep r
                # The test rep is rep r.
                train_reps = list(range(1,num_reps+1))
                # This pop function returns the popped element and removes the rth element from the training list.
                test_rep = [train_reps.pop(r)]

                # Get a list of the windows that belong to the test set and training set.
                testing_ids = subject_rep.numpy() == test_rep
                training_ids = [not elem for elem in testing_ids]

                # Partition the training and testing features / labels
                train_features = features[training_ids,:]
                test_features = features[testing_ids,:]
                
                train_class = subject_class[training_ids]
                test_class  = subject_class[testing_ids]

                mdl = LinearDiscriminantAnalysis()
                # If you'd rather use SVM, you can do so using this code instead!
                # mdl = SVC(kernel='linear')
                mdl.fit(train_features, train_class)
                predictions = mdl.predict(test_features)

                within_subject_results[s,r,f] = np.sum(predictions == test_class.numpy())/test_features.shape[0] * 100

    # I am planning on using the github readme file to keep track of the results of different pipelines, so let's output the results in markup format
    subject_accuracy = np.mean(within_subject_results, axis=1)
    featureset_accuracy = np.mean(subject_accuracy,axis=0)
    featureset_std      = np.std(subject_accuracy,axis=0)
    print("| Subject | ", end='')
    for f in featuresets:
        print(" {} |".format(f), end='')
    print("")
    for c in range(num_featuresets+1):
        print("| --- ",end='')
    print("|")

    for s in range(num_subjects):
        print("| S{} |".format(str(s)), end='')
        for f in range(num_featuresets):
            print(" {} |".format(str(subject_accuracy[s,f])),end='')
        print("")
    
    print("| Mean | ",end="")
    for f in range(num_featuresets):
        print(" {} |".format(str(featureset_accuracy[f])), end="")
    print("")

    print("| STD | ",end="")
    for f in range(num_featuresets):
        print(" {} |".format(str(featureset_std[f])), end="")
    print("")

    print("| TIME (ms) | ",end="")
    for f in range(num_featuresets):
        print(" {} |".format(str(featureset_times[f]*1000)), end="")
    print("")


    np.save("Results/withinsubject_handcrafted.npy", within_subject_results)