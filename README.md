# Multimodal Inductive Transfer Learning for Detection of Alzheimer’s Dementia and its Severity

> [Multimodal Inductive Transfer Learning for Detection of Alzheimer’s Dementia and its Severity](https://github.com/wazeerzulfikar/ad-mmse/blob/master)
> Utkarsh Sarawgi\*, Wazeer Zulfikar\*, Nouran Soliman, Pattie Maes
> To appear in INTERSPEECH 2020

This work was also submitted to the [Alzheimer's Dementia Recognition through Spontaneous Speech (ADReSS) challenge](http://www.homepages.ed.ac.uk/sluzfil/ADReSS/)

## Abstract

Alzheimer's disease is estimated to affect around 50 million people worldwide and is rising rapidly, with a global economic burden of nearly a trillion dollars. This calls for scalable, cost-effective, and robust methods for detection of Alzheimer's dementia (AD). We present a novel architecture that leverages acoustic, cognitive, and linguistic features to form a multimodal ensemble system. It uses specialized artificial neural networks with temporal characteristics to detect AD and its severity, which is reflected through Mini-Mental State Exam (MMSE) scores. We first evaluate it on the ADReSS challenge dataset, which is a subject-independent and balanced dataset matched for age and gender to mitigate biases, and is available through DementiaBank. Our system achieves state-of-the-art test accuracy, precision, recall, and F1-score of 83.3\% each for AD classification, and state-of-the-art test root mean squared error (RMSE) of 4.60 for MMSE score regression. To the best of our knowledge, the system further achieves state-of-the-art AD classification accuracy of 88.0\% when evaluated on the full benchmark DementiaBank Pitt database. Our work highlights the applicability and transferability of spontaneous speech to produce a robust inductive transfer learning model, and demonstrates generalizability through a task-agnostic feature-space.

## Highlights

- **Simple:** Fast and elegant models which when ensembled produces competitive results in multiple tasks

- **Multimodal:** Uses Disfluency, Acoustic, and Inter-vention features with voting for a robust model

- **Strong:** Our ensemble model achieves _83.3\%_ classification accuracy and _4.60_ rmse for MMSE score regression in ADReSS

- **Robust:** Balanced dataset for gender and age by ADReSS ensures rigid testing

## Main Results

### ADReSS test set

| Model                 | Accuracy | Precision | Recall | F1-Score | RMSE (MMSE\*)|
|-----------------------|----------|-----------|--------|----------|--------------|
| Luz et al.            | 0.75     | **0.83**      | 0.62   | 0.71     | 5.21         |
| Sarawgi et al. (Ours) | **0.83**     | **0.83**     | **0.83**   | **0.83**     | **4.60**         |

\* Mini Mental State Exam scores

### Full DementiaBank corpus

| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Fraser et al.         | 0.82     | -         | -      | -        |
| Masrani               | 0.85     | -         | -      | 0.85     |
| Kong et al.           | 0.87     | 0.86      | **0.91**   | **0.88**     |
| Sarawgi et al. (Ours) | **0.88**     | **0.92**      | 0.82   | **0.88**     |

*Above results for DementiaBank are using 10 fold cross validation*

### Individual modal performance on ADReSS test set

![roc](https://github.com/wazeerzulfikar/ad-mmse/blob/master/img/roc.png)

## Usage 

### Dataset Download

Request access from [DementiaBank](https://dementia.talkbank.org/)

### Setup

1. Install dependencies using `pip install -r requirements.txt`
2. Install and setup OpenSmile for Compare features extraction
3. Extract compare features

### Run

Set config parameters in `main.py` and run `python main.py`

### Model Architecture
We use an Ensemble model of (1) Disfluency, (2) Acoustic, and (3) Inter-ventions models for AD classification.
Then (4) Regression module is added at the top of the Ensemble for MMSE regression.

![model architecture](https://github.com/wazeerzulfikar/ad-mmse/blob/master/img/model_final.jpeg)

## License

## Citation

If you find this project useful for your research, please cite using the following entry

### Compare features extraction

1. Download the [opensmile](https://www.audeering.com/opensmile/) toolkit.
2. `tar -zxvf openSMILE-2.x.x.tar.gz`
3. `cd openSMILE-2.x.x`
4. `bash autogen.sh`
5. `make -j4`
6. `make`
7. `make install`
