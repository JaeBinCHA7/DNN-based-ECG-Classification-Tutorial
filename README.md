# ECG Multi-class classification tutorial 
This repository is created to experience ECG arrhythmia classification. 
It is a multi-class classification problem that aims to diagnose one of five types of arrhythmias. 
For convenience, an ipynb version of the tutorial is also provided.

## Update:  
* **2024.11.12** Upload codes

## Requirements 
This repo is tested with Ubuntu 22.04, PyTorch 2.0.5, Python3.10, and CUDA12.4
```
pip install -r requirements.txt    
```
## Dataset Installation 
MIT-BIH Database [Link](https://archive.physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm)

The MIT-BIH Arrhythmia Database is a well-known dataset used for the study of ECG arrhythmias. 
It contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects.

For experiments, we split the dataset into training, validation, and test sets with a ratio of 6:2:2.
We saved the signal data and labels as numpy files. 
You can use the numpy files in the [dataset](https://github.com/JaeBinCHA7/DNN-based-ECG-Classification-Tutorial/tree/main/dataset).

## Getting started    
1. Install the necessary libraries.   
2. Set directory paths for your dataset. ([options.py](https://github.com/JaeBinCHA7/DNN-based-ECG-Classification-Tutorial/blob/main/options.py))    
3. Run [train.py](https://github.com/JaeBinCHA7/DNN-based-ECG-Classification-Tutorial/blob/main/train.py)

## Evaluation 
To evaluate the trained model on the test dataset, use [test.py](https://github.com/JaeBinCHA7/DNN-based-ECG-Classification-Tutorial/blob/main/test.py)
```
python test.py
```

## References   
[1] Moody, George B., and Roger G. Mark. "The impact of the MIT-BIH arrhythmia database." IEEE engineering in medicine and biology magazine 20.3 (2001): 45-50.

## Contact   
E-mail: jbcha7@yonsei.ac.kr
