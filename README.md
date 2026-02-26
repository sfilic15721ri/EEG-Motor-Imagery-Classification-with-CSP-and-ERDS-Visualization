# EEG-Motor-Imagery-Classification-with-CSP-and-ERDS-Visualization
This project demonstrates the analysis of EEG data from motor imagery tasks using MNE-Python. The workflow includes preprocessing, classification using CSP + LDA, and visualization of Event-Related Desynchronization/Synchronization (ERDS).

Features

EEG Data Loading: Fetches data from the EEGBCI dataset for a specific subject and runs.

Preprocessing:

Channel renaming and standard 10-05 montage setup

Band-pass filtering between 7–30 Hz

Epoch Extraction: Epochs extracted between 1–4 seconds of task performance

Classification:

CSP (Common Spatial Patterns) to enhance discriminative spatial patterns

Linear Discriminant Analysis (LDA) classifier

5-fold cross-validation with average accuracy output

ERDS Visualization:

Time–frequency analysis using Morlet wavelets

Baseline correction and percentage change visualization

Plot example for a single channel (C3)

Usage

Install required packages:

pip install mne numpy matplotlib scikit-learn

Run the script to load EEG data, train the classifier, and visualize ERDS.

The script prints average classification accuracy and shows the ERDS plot for channel C3.
