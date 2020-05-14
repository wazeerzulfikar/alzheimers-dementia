'''
Counting the number of Investigator dialogues and using a hard threshold to classify AD.
The thresholds are looped over 1 to 9 to find best one.
'''

import glob
import os

print('------- CC ------')

dataset_dir = 'ADReSS-IS2020-data/train/transcription/cc/'
files = sorted(glob.glob(os.path.join(dataset_dir, '*.cha')))
all_inv_counts_cc = []
for filename in files:
    inv_count = 0
    with open(filename, 'r') as f:
        content = f.read().split('\n')
        speaker_cc = []
        uh_count = 0

        for c in content:
            if 'INV' in c:
                speaker_cc.append('INV')
            if 'PAR' in c:
                speaker_cc.append('PAR')
                uh_count+=c.count('uh')
        
        PAR_first_index = speaker_cc.index('PAR')
        PAR_last_index = len(speaker_cc) - speaker_cc[::-1].index('PAR') - 1 
        speaker_cc = speaker_cc[PAR_first_index:PAR_last_index]
        inv_count = speaker_cc.count('INV')
    all_inv_counts_cc.append(inv_count)
    print('{} has {} INVs'.format(filename.split('/')[-1], inv_count))
    # print('{} has {} uhs'.format(filename.split('/')[-1], uh_count))

print('------- CD ------')
dataset_dir = 'ADReSS-IS2020-data/train/transcription/cd/'
files = sorted(glob.glob(os.path.join(dataset_dir, '*.cha')))
all_inv_counts_cd = []
for filename in files:
    inv_count = 0
    with open(filename, 'r') as f:
        content = f.read().split('\n')
        speaker_cd = []
        uh_count = 0

        for c in content:
            if 'INV' in c:
                speaker_cd.append('INV')
            if 'PAR' in c:
                speaker_cd.append('PAR')
                uh_count+=c.count('uh')

        PAR_first_index = speaker_cd.index('PAR')
        PAR_last_index = len(speaker_cd) - speaker_cd[::-1].index('PAR') - 1 
        speaker_cd = speaker_cd[PAR_first_index:PAR_last_index]
        inv_count = speaker_cd.count('INV')
    all_inv_counts_cd.append(inv_count)
    print('{} has {} INVs'.format(filename.split('/')[-1], inv_count))
    # print('{} has {} uhs'.format(filename.split('/')[-1], uh_count))

for threshold in range(10):
    cc_pred = 0
    cd_pred = 0
    for cc, cd in zip(all_inv_counts_cc, all_inv_counts_cd):
        if cc > threshold:
            cc_pred+=1
        if cd > threshold:
            cd_pred+=1
    print('Threshold of {} Investigator dialogues'.format(threshold))
    print('Diagnosed {} healthy people'.format(cc_pred))

    print('Diagnosed {} AD people'.format(cd_pred))


    precision = cd_pred / (cc_pred + cd_pred)
    recall = cd_pred / ((54-cd_pred) + cd_pred)
    accuracy = (54-cc_pred+cd_pred)/108
    f1_score = (2*precision*recall)/(precision+recall)
    print('Accuracy {:.3f} '.format(accuracy))
    print('F1 score {:.3f}'.format(f1_score))
    print('Precision {:.3f}'.format(precision))
    print('Recall {:.3f}'.format(recall))

    print('----'*50)