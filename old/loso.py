import numpy as np
import glob as glob

files = glob.glob('models/bagging_loocv_1/*.npy')

preds = []
for f in files: 
	preds.append(np.load(f))

preds = np.stack(preds, axis=1)
print(preds.shape)

preds = [max(set(i), key = list(i).count) for i in preds]

print(np.mean(preds))