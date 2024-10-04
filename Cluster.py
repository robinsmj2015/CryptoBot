#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:15:14 2022

@author: robinson
"""


from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
import File_Utils

window = 1
cb_num = 27
file_num = 36
look_aheads = 1
markov_number = 1
bool_roi = True
numerical = True
time_mode = '1st'
interval = '5m'
crypto = 'ADA'
price_comp = True
tern = False
splits = 10
features = ['EMA10', 'SMA10', 'EMA20', 'SMA20', 'EMA50', 'SMA50', 'EMA100', 'SMA100', \
  'EMA200', 'SMA200', 'UO', 'AO', 'ADX', 'RSI', 'Mom', 'HullMA', 'STOCH.K', 'VWMA', 'Stoch.RSI', \
      'BBP', 'MACD', 'Ichimoku', 'CCI', 'W%R', 'EMA30', 'SMA30']


all_x, all_y, all_roi, all_numeric_x, cols = File_Utils.load_training_data(time_mode, \
                                                      interval, crypto, features, \
                                                        file_num, look_aheads, \
                                                            markov_number, cb_num, window, tern, price_comp)

    
piv = []
for i in range(51, 82):
    piv.append(i)
all_numeric_x = np.delete(all_numeric_x, [4, 7, 8, 10, 14, 15, 19, 21, 86, 22, 24, 28, 30, 45, 47, 49, 82, 87, 88, 89, 90] + piv, 1) # 47, 49 ind rec and/or 0, 1, 2 overall rec
    
labels_true = all_y
X = all_numeric_x
# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print(
    "Adjusted Mutual Information: %0.3f"
    % metrics.adjusted_mutual_info_score(labels_true, labels)
)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()