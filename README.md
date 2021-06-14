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
