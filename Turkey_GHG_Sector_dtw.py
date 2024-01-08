# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:09:24 2024

@author: Huawei
"""
# Import the appropriate libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn import preprocessing
import os
from sklearn.metrics import silhouette_score

# Import the data
directory = 'C:\\Users\\Huawei\\Masaüstü\\Hacettepe doktora\\EMÜ 737 Veri madenciliği\\Project\\Inputs'  
filename = 'Sector_based_GHG.csv'
file_path = os.path.join(directory, filename)
Turkey_GHG_df = pd.read_csv(file_path)
# Scale the data
data = Turkey_GHG_df.iloc[:, 1:11]
scaler = preprocessing.StandardScaler().fit(data)
data_scaled = scaler.transform(data)

# Dynamic time warping (DTW)
def dtw_basic(x, y):
    n, m = len(x), len(y)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(1, n+1):
        dtw_matrix[i, 0] = np.inf
    for i in range(1, m+1):
        dtw_matrix[0, i] = np.inf
    dtw_matrix[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(x[i-1] - y[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                         dtw_matrix[i, j-1],    # deletion
                                         dtw_matrix[i-1, j-1])  # match
    return dtw_matrix[n, m]
# Compute the DTW distance matrix
n_samples = data_scaled.shape[0]
dtw_distance_matrix = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(i+1, n_samples):
        dtw_distance = dtw_basic(data_scaled[i], data_scaled[j])
        dtw_distance_matrix[i, j] = dtw_distance
        dtw_distance_matrix[j, i] = dtw_distance
        
# Perform hierarchical clustering using AgglomerativeClustering with the precomputed DTW distance matrix
clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', distance_threshold=0)
clustering.fit(dtw_distance_matrix)
# Create linkage matrix 
counts = np.zeros(clustering.children_.shape[0])
for i, merge in enumerate(clustering.children_):
    current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1  # Leaf node
        else:
            current_count += counts[child_idx - n_samples]
    counts[i] = current_count
linkage_matrix = np.column_stack([clustering.children_, clustering.distances_, counts]).astype(float)
# Plot the dendrogram
plt.figure(figsize=(10, 10))  
dendrogram(linkage_matrix, orientation='top', distance_sort='descending', show_leaf_counts=True, leaf_rotation=90, leaf_font_size=8, labels= range(1,34))
plt.title('Hierarchical Clustering Dendrogram (DTW)')
plt.xlabel('Sample index or (cluster size)')
plt.ylabel('Distance')
plt.show()
        
# Grid Search to determine best k and the linkage method by using silhouette score
param_grid = {'n_clusters': [2, 3, 4, 5], 'linkage': ['complete', 'average', 'single']}
# List to store the results
results = []
# Iterate over parameter grid
for n_clusters in param_grid['n_clusters']:
    for linkage in param_grid['linkage']:
        # Create AgglomerativeClustering model with current parameters
        model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage)
        # Fit the model and get labels
        labels = model.fit_predict(dtw_distance_matrix)
        # Calculate silhouette score
        silhouette = silhouette_score(data_scaled, labels)
        # Append results to the list
        results.append({'n_clusters': n_clusters, 'linkage': linkage, 'silhouette_score': silhouette})
# Convert the results list to a DataFrame and sort it by silhouette_score
results_df = pd.DataFrame(results).sort_values(by='silhouette_score', ascending=False)
results_df.reset_index(drop=True, inplace=True)
results_df['rank'] = results_df.index + 1
# Display the results DataFrame with ranking
print(results_df)

