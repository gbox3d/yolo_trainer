#%%
import sys
import numpy as np
sys.path.append('../../detector/modules/yolov5')
# from utils.plots import plot_results 
import glob
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


#%%
save_dir = '/home/gbox3d/work/dataset/madang/train_jobs/exp2'
s = ['Box', 'Objectness', 'Classification', 'Precision', 'Recall',
        'val Box', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']

files = list(Path(save_dir).glob('results*.txt'))
assert len(files), 'No results.txt files found in %s, nothing to plot.' % os.path.abspath(save_dir)

#%%
start=0
stop=0
labels=()
for fi, f in enumerate(files):
    try:
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        for i in range(10):
            y = results[i, x]
            if i in [0, 1, 2, 5, 6, 7]:
                y[y == 0] = np.nan  # don't show zero loss values
                # y /= y[0]  # normalize
            label = labels[fi] if len(labels) else f.stem
            # print(y)
            plt.plot(x, y, marker='.', label=label, linewidth=2, markersize=8)
    except Exception as e:
        print('Warning: Plotting error for %s; %s' % (f, e))

# %%
