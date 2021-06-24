# EMG_Tutorials

In order to run the tutorial, first get the dataset from google drive: 

Next, run the main_construct_dataset function.  This performs windowing of the dataset and stores the dataset as .npy files.

These files can be used in the subsequent tutorial scripts.

1. Within Subject Handcrafted Feature Pipeline:
This pipeline performs gesture recognition using statistical features (for instance, mean absolute value, zero crossings, slope sign change, waveform length) and statistical classifiers (linear discriminant analysis, support vector machine).
In progress

2. Within Subject Deep learning Pipeline Using Convolutional Neural Networks
In progress


3. Between Subject Handcrafted Feature Pipeline:
This shows the degradation experienced when users wish to use an sEMG gesture recognition system but do not provide data themselves through a recent acquisition protocol (See Saponas et al.).
In progress


4. Between Subject Handcrafted Features Pipeline with Projection Techniques (Canonical Correlation Analysis):
Prior to 2020, the state-of-the-art technique for achieving high performance for between subject gesture recognition relied on canonical correlation analysis (See Khushaba et al 2015).
In progress


5. Subject-Independent Adaptive Domain Adversarial Neural Network:
This technique builds a model that is well suited for many subjects.  This was a breakthrough towards between subject gesture recognition, but at this point a full acquisition protocol was still required by all end users (Cote-Allard et al 2020).
In progress


6. Between Subject Adaptive Domain Adversarial Neural Network:
This technique builds off the subject-independent adaptive domain adversarial neural network but has a mechanism to adapt to a subject that only provides a single repetition (Campbell et al 2021).
In progress




## Within Subject Results

| Subject |  TD | TDPSD | LSF4 | LSF9 |  CNN |
| --- | --- | --- | --- | --- | --- |
| S0 | 96.28059405065319 | 98.56654908559705 | 98.35276652018267 | 98.7307863078328 | 99.41452297815479 |
| S1 | 92.51591477500472 | 95.14179058412597 | 95.20592566635236 | 96.12663088881233 | 91.98451775994974 |
| S2 | 79.98360633675084 | 87.80965845505101 | 87.40853755760294 | 88.49376938758729 | 82.84690424656084 |
| S3 | 94.27212213215147 | 97.81476234326948 | 96.9501681444503 | 97.38578874842342 | 98.34340933983604 |
| S4 | 90.2387268463126 | 92.31312443007117 | 92.97773680298462 | 94.35339496396251 | 92.3039876748737 |
| S5 | 84.74000178134908 | 88.52049651157299 | 88.1928249760174 | 88.39310135008016 | 88.94057710914986 |
| S6 | 68.76059863812824 | 73.57149850658146 | 71.9219018079592 | 74.72747812834588 | 71.05533519526878 |
| S7 | 82.91067080347233 | 88.69695863046176 | 87.75371866558746 | 87.96901335158972 | 89.80586879149514 |
| S8 | 89.0038670320066 | 92.16268090014776 | 92.7841280938596 | 93.19802624276724 | 90.88853727483391 |
| S9 | 92.24744750289952 | 95.7731074458428 | 95.40857665655058 | 96.16485755785428 | 95.6602616884948 |
| Mean |  87.09535498987285 | 91.03706268927213 | 90.69562848915473 | 91.55428469272556 |  90.12439220586175 |
| STD |  7.842983256183912 | 6.870777987868703 | 7.266719880825716 | 6.734721042939276 |  7.81916163111671 |
| TIME (ms) |  0.03975517799803106 | 0.07131806242423407 | 1.3518062178069783 | 1.4118450304628045 | -- |
