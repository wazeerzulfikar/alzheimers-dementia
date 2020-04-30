data_path = "/Users/nouransoliman/Desktop/nlp project/data/train/transcription"

import os
import re
import operator
import pickle 

#.replace(".", "").replace("?", "")
def clean_file(lines):
  return re.sub(r'[0-9]+[_][0-9]+', '', lines.replace("*INV:", "").replace("*PAR:", "")).strip().replace("\x15", "").replace("\n", "").replace("\t", " ").replace("[+ ", "[+").replace("[* ", "[*").replace("[: ", "[:").replace(" .", "").replace("'s", "").replace(" ?", "").replace(" !", "").replace(" ]", "]").lower()

def extra_clean(lines):
  lines = lines.replace("[+exc]", "")
  lines = lines.replace("[+gram]", "")
  lines = lines.replace("[+es]", "")
  lines = re.sub(r'[&][=]*[a-z]+', '', lines) #remove all &=text
  lines = re.sub(r'\[[*][a-z]:[a-z][-|a-z]*\]', '', lines) #remove all [*char:char(s)]
  lines = re.sub(r'[^A-Za-z0-9\s_]+', '', lines) #remove all remaining symbols except underscore
  lines = re.sub(r'[_]', ' ', lines) #replace underscore with space
  return lines

def get_tot_cnt(list_datacc):
  cnt_cc = {}
  for i in range(len(list_datacc)): #file index
    for j in range(len(list_datacc[i])): #word index in ith file
      if list_datacc[i][j] in cnt_cc.keys():
        cnt_cc[list_datacc[i][j]] += 1
      else:
        cnt_cc[list_datacc[i][j]] = 1
  return sorted(cnt_cc.items(), key=operator.itemgetter(1), reverse=True)

def get_file_cnt(file):
  cnt = {}
  for word in file: #word index in file
    if word:
      if word in cnt.keys():
        cnt[word] += 1
      else:
        cnt[word] = 1
  return sorted(cnt.items(), key=operator.itemgetter(1), reverse=True)

def get_pauses_cnt(file):
  cnt = 0
  pauses_list = []
  pauses = re.findall(r'&[a-z]+', file) #find all utterances
  cnt += len(pauses)
  pauses_list.append(pauses)

  pauses = re.findall(r'<[a-z_\s]+>', file) #find all <text>
  cnt += len(pauses)
  pauses_list.append(pauses)

  pauses = re.findall(r'\[/+\]', file) #find all [/+]
  cnt += len(pauses)
  pauses_list.append(pauses)

  pauses = re.findall(r'\([\.]+\)', file) #find all (.*)
  cnt += len(pauses)
  pauses_list.append(pauses)

  pauses = re.findall(r'\+[\.]+', file) #find all +...
  cnt += len(pauses)
  pauses_list.append(pauses)

  pauses = re.findall(r'[m]*hm', file) #find all mhm or hm
  cnt += len(pauses)
  pauses_list.append(pauses)

  pauses = re.findall(r'\[:[a-z_\s]+\]', file) #find all [:word]
  cnt += len(pauses)
  pauses_list.append(pauses)

  pauses = re.findall(r'[a-z]*\([a-z]+\)[a-z]*', file) #find all wor(d)
  cnt += len(pauses)
  pauses_list.append(pauses)

  temp = re.sub(r'\[[*][a-z]:[a-z][-|a-z]*\]', '', file)
  pauses = re.findall(r'[a-z]+:[a-z]+', temp) #find all w:ord
  cnt += len(pauses)
  pauses_list.append(pauses)

  #print(pauses_list)
  return cnt

def get_intervention_cnt(file):
  return file.count("*INV:")

def get_file_feats(orig_file, intervention):
  dictn = {}
  extra_cleaned = extra_clean(orig_file).split(" ")
  dictn["vocab_size"] = len(get_file_cnt(extra_cleaned))
  dictn["conversation_size"] = len(extra_cleaned) - extra_cleaned.count('')
  dictn["intervention_cnt"] = intervention
  dictn["pauses_cnt"] = get_pauses_cnt(orig_file)
  arr = [dictn["vocab_size"], dictn["conversation_size"], dictn["intervention_cnt"], dictn["pauses_cnt"]]
  return [dictn, arr]

directcc = data_path+"/cc_txt/"
directcd = data_path+"/cd_txt/"

datacc = []
datacd = []
dataall = []

intervention_cc = []
intervention_cd = []
intervention_all = []

filenames_all = []

for filename in os.listdir(directcc):
    if filename.endswith(".txt") and filename != "cleaned_cc.txt":
      f = open(directcc+filename)
      lines = f.read()
      inter = get_intervention_cnt(lines)
      intervention_cc.append(inter)
      intervention_all.append(inter)
      st = clean_file(lines)
      datacc.append(st)
      dataall.append(st)
      filenames_all.append(filename)

for filename in os.listdir(directcd):
    if filename.endswith(".txt") and filename != "cleaned_cd.txt":
      f = open(directcd+filename)
      lines = f.read()
      inter = get_intervention_cnt(lines)
      intervention_cd.append(inter)
      intervention_all.append(inter)
      st = clean_file(lines)
      datacd.append(st)
      dataall.append(st)
      filenames_all.append(filename)

print(len(datacc), len(datacd))

# list_datacc = [i.split(" ") for i in datacc] 
# list_datacd = [i.split(" ") for i in datacd] 
# list_dataall = [i.split(" ") for i in dataall] 

# print(len(list_dataall))

# ind = 72

# for i, file in enumerate(dataall):
#   ex = extra_clean(file)
#   c = get_file_cnt(ex.split(" "))
#   if i == ind:
#     print(filenames_all[i], i)
#     print(file)
#     print(ex)
#     # for w in c:
#     #   print(w)
#     print(get_file_feats(file, intervention_all[i])[0])

#get_file_feats returns parsing data in the form of dictionary (index 0) and array (index 1), set the index to use whichever you prefer
features = []
for i, file in enumerate(dataall):
  f = get_file_feats(file, intervention_all[i])
  features.append(f[0]) #currently appending the dictionary form
  #print(filenames_all[i], i, f[0], f[1])

with open(data_path+'/count_features.pkl', "wb") as f:
  pickle.dump(features,f)
  f.close()

# with open(data_path+'/count_features.pkl', "rb") as f:
#   new_feats = pickle.load(f)
#   f.close()

# print(len(new_feats))

print("Done!")

