# Multimodal Inductive Transfer Learning for Detection of Alzheimer’s Dementia and its Severity

Code for the paper - link

This work was also submitted to the [Alzheimer's Dementia Recognition through Spontaneous Speech (ADReSS) challenge](http://www.homepages.ed.ac.uk/sluzfil/ADReSS/)

## Dataset Download

Request access from [DementiaBank](https://dementia.talkbank.org/)

## Setup

1. Install dependencies using `pip install -r requirements.txt`
2. Install and setup OpenSmile for Compare features extraction
3. Extract compare features

## Run

Set config parameters and run `python main.py`

## Results

### ADreSS Test

| Model                 | Accuracy | Precision | Recall | F1-Score | RMSE (MMSE\*)|
|-----------------------|----------|-----------|--------|----------|--------------|
| Luz et al.            | 0.75     | **0.83**      | 0.62   | 0.71     | 5.21         |
| Sarawgi et al. (Ours) | **0.83**     | **0.83**     | **0.83**   | **0.83**     | **4.60**         |

\* Mini Mental State Exam scores

### DementiaBank Corpus

| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Fraser et al.         | 0.82     | -         | -      | -        |
| Masrani               | 0.85     | -         | -      | 0.85     |
| Kong et al.           | 0.87     | 0.86      | **0.91**   | **0.88**     |
| Sarawgi et al. (Ours) | **0.88**     | **0.92**      | 0.82   | **0.88**     |

*Above results are using 10 fold cross validation*

## Model Architecture
We use an Ensemble model of (1) Disfluency, (2) Acoustic, and (3) Inter-ventions models for AD classification.
Then (4) Regression module is added at the top of the Ensemble for MMSE regression.

![model architecture](https://github.com/wazeerzulfikar/ad-mmse/blob/master/img/model_final.jpeg)

## Individual modal performance on ADReSS

![roc](https://github.com/wazeerzulfikar/ad-mmse/blob/master/img/roc.png)

### Compare features extraction

1. Download the [opensmile](https://www.audeering.com/opensmile/) toolkit.
2. `tar -zxvf openSMILE-2.x.x.tar.gz`
3. `cd openSMILE-2.x.x`
4. `bash autogen.sh`
5. `make -j4`
6. `make`
7. `make install`
