# CIFAR-10-classification
A demo code for classification task of CIFAR-10 dataset

# Environment Requirement
python == 3.5.2

tensorflow-gpu == 1.12.0

keras == 2.2.4

# Guide
test.py divides CIFAR-10 dataset into two categories (i.e., 'cat' and 'no cat').

For training set, the numbers of samples of 'cat' and 'no cat' are both 5000;

For test set, the numbers of samples of the two categories are both 1000.

'no cat' samples are uniformly randomly selected from the samples of categories which are not 'cat'.

# Command
python test.py
