'''
Util functions for reading and extracting data and other stuff
'''
from pathlib import Path
import contextlib
import re
import wave
from mutagen.mp3 import MP3

import numpy as np

################# Utils #################

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name] 

def create_directories(config):
    model_dir = Path(config.model_dir)
    for m in config.model_types:
        model_dir.joinpath(m).mkdir(parents=True, exist_ok=True)

################# PAUSE FEATURES #################

def clean_file(lines):
    return re.sub(r'[0-9]+[_][0-9]+', '', lines.replace("*INV:", "").replace("*PAR:", "")).strip().replace("\x15", "").replace("\n", "").replace("\t", " ").replace("[+ ", "[+").replace("[* ", "[*").replace("[: ", "[:").replace(" .", "").replace("'s", "").replace(" ?", "").replace(" !", "").replace(" ]", "]").lower()

def extra_clean(lines):
    lines = clean_file(lines)
    lines = lines.replace("[+exc]", "")
    lines = lines.replace("[+gram]", "")
    lines = lines.replace("[+es]", "")
    lines = re.sub(r'[&][=]*[a-z]+', '', lines) #remove all &=text
    lines = re.sub(r'\[[*][a-z]:[a-z][-|a-z]*\]', '', lines) #remove all [*char:char(s)]
    lines = re.sub(r'[^A-Za-z0-9\s_]+', '', lines) #remove all remaining symbols except underscore
    lines = re.sub(r'[_]', ' ', lines) #replace underscore with space
    return lines

def words_count(content):
    extra_cleaned = extra_clean(content).split(" ")
    return len(extra_cleaned) - extra_cleaned.count('')

def get_pauses_cnt(content):
    content = clean_file(content)

    cnt = 0
    pauses_list = []
    pauses = re.findall(r'&[a-z]+', content) #find all utterances
    cnt += len(pauses)
    pauses_list.append(len(pauses))

    pauses = re.findall(r'<[a-z_\s]+>', content) #find all <text>
    cnt += len(pauses)
    pauses_list.append(len(pauses))

    pauses = re.findall(r'\[/+\]', content) #find all [/+]
    cnt += len(pauses)
    pauses_list.append(len(pauses))

    pauses = re.findall(r'\([\.]+\)', content) #find all (.*)
    cnt += len(pauses)
    pauses_list.append(len(pauses))

    pauses = re.findall(r'\+[\.]+', content) #find all +...
    cnt += len(pauses)
    pauses_list.append(len(pauses))

    pauses = re.findall(r'[m]*hm', content) #find all mhm or hm
    cnt += len(pauses)
    pauses_list.append(len(pauses))

    pauses = re.findall(r'\[:[a-z_\s]+\]', content) #find all [:word]
    cnt += len(pauses)
    pauses_list.append(len(pauses))

    pauses = re.findall(r'[a-z]*\([a-z]+\)[a-z]*', content) #find all wor(d)
    cnt += len(pauses)
    pauses_list.append(len(pauses))

    temp = re.sub(r'\[[*][a-z]:[a-z][-|a-z]*\]', '', content)
    pauses = re.findall(r'[a-z]+:[a-z]+', temp) #find all w:ord
    cnt += len(pauses)
    pauses_list.append(len(pauses))

    return np.array(pauses_list)


################# INTERVENTION FEATURES #################

def get_n_interventions(content):
    content = content.split('\n')
    speakers = []

    for c in content:
        if 'INV' in c:
          speakers.append('INV')
        if 'PAR' in c:
          speakers.append('PAR')

    PAR_first_index = speakers.index('PAR')
    PAR_last_index = len(speakers) - speakers[::-1].index('PAR') - 1 
    speakers = speakers[PAR_first_index:PAR_last_index]
    return speakers.count('INV')


################# SPECTOGRAM FEATURES #################

def read_spectogram():
    return

################# REGRESSION FEATURES #################

def get_regression_values(metadata_filename):
    values = []
    with open(metadata_filename, 'r') as f:
        content = f.readlines()[1:]
        for idx, line in enumerate(content):
            token = line.split('; ')[-1].strip('\n')
            if token!='NA':  values.append(int(token))
            else:   values.append(30) # NA fill value

    return values

def get_classification_values(metadata_filename):
    values = []
    with open(metadata_filename, 'r') as f:
        content = f.readlines()[1:]
        for idx, line in enumerate(content):
            token = line.split('; ')[-2].strip('\n')
            if token!='NA':  values.append(int(token))
            else:   values.append(30) # NA fill value

    return values

def get_audio_length(filename):
    with contextlib.closing(wave.open(filename,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration

def get_mp3_audio_length(filename):
    audio = MP3(filename)
    duration = audio.info.length
    return duration



