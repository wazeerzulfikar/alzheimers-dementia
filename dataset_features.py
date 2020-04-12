import cv2
import numpy as np

import dataset_utils

def get_pause_features(transcription_filename, audio_filename, audio_length_normalization=10):
    '''
    Pause features include word rate, pause rate of various kinds of pauses and utterances, and intervention rate
    '''
    audio_len =  dataset_utils.get_audio_length(audio_filename)/audio_length_normalization

    with open(transcription_filename, 'r') as f:
        content = f.read()
        word_rate = dataset_utils.words_count(content) / (50 * audio_len)
        pause_rates = dataset_utils.get_pauses_cnt(content) / audio_len
        inv_rate = dataset_utils.get_n_interventions(content) / audio_len

    pause_features = np.concatenate(([inv_rate], pause_rates, [word_rate]), axis=-1)

    return pause_features


def get_intervention_features(transcription_filename, max_length=40):
    '''
    Intervention features include one hot encoded sequence of speaker (par or inv) in the conversation
    '''
    speaker_dict = {
        'INV': [0 ,0 , 1],
        'PAR': [0, 1, 0], 
        'padding': [1, 0, 0]
    }

    with open(transcription_filename, 'r') as f:
        content = f.read()
        content = content.split('\n')
        speakers = []

        for c in content:
            if 'INV' in c:
              speakers.append('INV')
            if 'PAR' in c:
              speakers.append('PAR')

        PAR_first_index = speakers.index('PAR')
        PAR_last_index = len(speakers) - speakers[::-1].index('PAR') - 1 
        intervention_features = speakers[PAR_first_index:PAR_last_index]

    intervention_features = list(map(lambda x: speaker_dict[x], intervention_features))

    if len(intervention_features) > max_length:
        intervention_features = intervention_features[:max_length]
    else:
        pad_length = max_length - len(intervention_features)
        intervention_features =intervention_features+[speaker_dict['padding']]*pad_length

    return intervention_features


def get_spectogram_features(spectogram_filename, output_size=(640, 480), normalize=True):
    '''
    Spectogram features include MFCC which has been pregenerated for the audio file 
    '''
    img = cv2.imread(spectogram_filename)
    img = cv2.resize(img, output_size)
    if normalize:
        img = img / 255.
    spectogram_features = img
    return spectogram_features

