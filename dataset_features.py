import cv2
import numpy as np
import csv
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

def feature_normalize(feature_data):
    mean = np.mean(feature_data, axis=0)
    std = np.std(feature_data, axis=0)
    N = feature_data.shape[0]
    S1 = np.sum(feature_data, axis=0)
    S2 = np.sum(feature_data ** 2, axis=0)
    mean=S1/N
    std=np.sqrt((N * S2 - (S1 * S1)) / (N * (N - 1)))
    mean = np.reshape(mean, [1, -1])
    std = np.reshape(std, [1, -1])
    feature_data=((feature_data-mean)/std)
    return feature_data

# def get_spectogram_features(spectogram_filename, output_size=(640, 480), normalize=True):
#     '''
#     Spectogram features include MFCC which has been pregenerated for the audio file
#     '''
#     img = cv2.imread(spectogram_filename)
#     img = cv2.resize(img, output_size)
#     if normalize:
#         # img = img / 255.
#         img = feature_normalize(img)
#     spectogram_features = img
#     return spectogram_features

def get_spectogram_features(spectogram_filename):
    '''
    Spectogram features include MFCC which has been pregenerated for the audio file
    '''
    mel = np.load(spectogram_filename)
    # mel = feature_normalize(mel)
    mel = np.expand_dims(mel, axis=-1)
    return mel

def get_old_spectogram_features(spectogram_filename):
    '''
    Spectogram features include MFCC which has been pregenerated for the audio file
    '''
    mel = cv2.imread(spectogram_filename)
    # mel = feature_normalize(mel)
    return mel


def get_sliced_spectogram_features(spectogram_filename, timesteps_per_slice=1000, normalize=True):
    '''
    Spectogram features include MFCC which has been pregenerated for the audio file
    '''
    mel = np.load(spectogram_filename)
    # mel = feature_normalize(mel)
    n_mels, timesteps = mel.shape
    # if normalize:
    #     mel_mean = np.mean(mel)
    #     mel_std = np.std(mel)
    #     mel = (mel - mel_mean) / mel_std
    # print(timesteps)
    # print(np.mean(mel), np.std(mel))
    # print('-'*20)
    mel_new = []
    for i in range(0, timesteps - timesteps_per_slice, 256):
        mel_new.append(mel[..., i:i+timesteps_per_slice])
    # mel = mel[..., :-(timesteps % timesteps_per_slice)]

    mel = np.reshape(mel_new, (-1, n_mels, timesteps_per_slice, 1))
    return mel

def get_compare_features(compare_filename):
    compare_features = []
    with open(compare_filename, 'r') as file:
        content = csv.reader(file)
        for row in content:
            compare_features = row
    compare_features_floats = [float(item) for item in compare_features[1:-1]]
    return compare_features_floats
