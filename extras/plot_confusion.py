import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


pred_files = sorted(glob.glob('plots/cm/pred*'))
preds = []
for f in pred_files:
    preds += list(np.load(f))

true_files = sorted(glob.glob('plots/cm/true*'))
labels = []
for f in true_files:
    labels += list(np.load(f))

report = precision_recall_fscore_support(labels, preds, average='binary')
accuracy = accuracy_score(labels, preds)

print('precision: {:.3f}, recall: {:.3f}, f1_score: {:.3f}, accuracy: {:.3f}'.format(report[0], report[1], report[2], accuracy))

# print(preds)
# print(labels)

# print(labels[55])

# a = np.load('models/bagging_loocv_1/inv.npy')
# b = np.load('models/bagging_loocv_1/pause.npy')
# c = np.load('models/bagging_loocv_1/compare.npy')

# p = np.stack((a,b,c), axis=1)

# p = [max(set(i), key = list(i).count) for i in p]

# print([i for i in range(len(p)) if p[i]==0])
print(preds)
print(labels)

array = [[51,3], 
        [5,49]]
array = [[20,4], 
        [4,20]]
# array = [[53,1], 
#         [0,54]]
df_cm = pd.DataFrame(array, index = [i for i in ['non-AD', 'AD']],
                  columns = [i for i in ['non-AD', 'AD']])
plt.figure(figsize = (11,9))
sn.set(font_scale=3.0)
sn.heatmap(df_cm, annot=True)
plt.xlabel('Predictions')
plt.ylabel('True Labels')

plt.savefig('plots/confusion_test.png')

#############################################################################
import glob
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# def get_stuff(model_type):
#     true_files = sorted(glob.glob('plots/*{}_true*.npy'.format(model_type)))
#     all_trues = []
#     for f in true_files:
#         t = np.load(f)
#         # print(np.shape(t))
#         all_trues += list(t)
#     print(len(all_trues))

#     pred_files = sorted(glob.glob('plots/*{}_pred*.npy'.format(model_type)))
#     all_preds = []
#     for f in pred_files:
#         t = np.load(f)
#         # print(np.shape(t))
#         all_preds += list(t)
#     print(len(all_preds))

#     fpr, tpr, thresholds = roc_curve(all_trues, all_preds, pos_label=1)
#     auc_score = auc(fpr, tpr)
#     print(model_type+" auc: ", auc_score)

#     return fpr, tpr, auc_score

# inv_fpr, inv_tpr, inv_score = get_stuff('inv')
# pause_fpr, pause_tpr, pause_score = get_stuff('pause')
# compare_fpr, compare_tpr, compare_score = get_stuff('compare')
# # print(fpr)
# # print(tpr)

# plt.figure()
# lw = 2
# plt.plot(pause_fpr, pause_tpr, color='red',
#          lw=lw, label='{} ROC curve (area = {:0.2f})'.format('Disfluency', pause_score))
# plt.plot(compare_fpr, compare_tpr, color='green',
#          lw=lw, label='{} ROC curve (area = {:0.2f})'.format('Acoustic', compare_score))
# plt.plot(inv_fpr, inv_tpr, color='darkorange',
#          lw=lw, label='{} ROC curve (area = {:0.2f})'.format('Interventions', inv_score))
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.savefig('plots/roc.png')
# plt.show()