import numpy as np 
import matplotlib.pyplot as plt 
import torch
import csv
import pandas as pd  
from scipy import stats as st

def kmeans_gpu(data, k, max_iters=100):
    data = data.to(device)
    centroids = data[torch.randperm(len(data))[:k]]

    for i in range(max_iters):
        print(f"Iteration: {i+1}")
        dist = torch.cdist(data, centroids) 
        clusters = torch.argmin(dist, dim=1)

        # update centroids 
        for j in range(k):
            centroids[j] = data[clusters == j].mean(dim=0)
    
    return clusters.cpu().numpy(), centroids

device = "cuda" if torch.cuda.is_available() else "cpu" 

df = pd.read_csv('500_events_noiseless.csv')
lf = pd.read_csv('labels_500_events_noiseless.csv')

columns = ['x[px]', 'y[px]', 't[s]']
xycolumns = ['x[px]', 'y[px]']
data_array = torch.Tensor(np.array(df[xycolumns]))
labels_array = torch.Tensor(np.array(lf['labels']))

# 30 is average number of photons per event
k = 500
clusters, centroids = kmeans_gpu(data_array, k, max_iters=5000) # clusters is the array of labels, centroids is the list of estimated centroids
centroids = centroids.numpy()
clusterFile = '500_events_noiseless_results_xy.csv'
with open(clusterFile, mode = 'w', newline='') as wfile: 
    writer = csv.writer(wfile)
    writer.writerow(['labels'])
    for item in clusters: 
                writer.writerow([item])

columns = ['x[px]', 'y[px]', 't[s]']
centroidsFile = '500_events_noiseless_centroids_xy.csv'
with open(centroidsFile, mode = 'w', newline = '') as wfile: 
    writer = csv.writer(wfile)
    writer.writerow(columns)
    writer.writerows(centroids)

# # print(clusters)
# print(centroids)
# #print(data_array)

# plt.scatter(data_array[:,0], data_array[:,1], c = clusters, cmap='magma', marker='+')
# plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='o', s=15)
# plt.title("Kmeans clustering 500 events x-y only")
# plt.xlabel("X coordinates")
# plt.ylabel("Y coordinates")
# plt.savefig('xy 1000 iterations k-means clustering 500 events.png')

# plt.figure()
# plt.scatter(data_array[:,2], data_array[:,0], c = clusters, cmap='magma', marker='+')
# plt.scatter(centroids[:,2], centroids[:,0], c='red', marker='o', s=15)
# plt.title('Kmeans clustering 500 noiseless events')
# plt.xlabel('Time (s)')
# plt.ylabel('X coordiantes')
# plt.savefig('xy 1000 iterations k-means space vs time 500 events.png')

def breaks(array): 
    '''This function takes an array, goes through it item by item, and returns the list of indices where the value changes'''
    value = array[0] 
    indices = []
    for index, ele in enumerate(array): 
        if ele != value:
            indices.append(index)
            value = array[index]
    indices.append(len(array))

    return indices

def sizes(indices): 
    '''This function takes a list of indices and calculates the number of items belonging to each value by taking the difference between 
    subsequent indices'''
    gaps = []
    prev=0
    for i in range(len(indices)):
        gaps.append(indices[i]-prev)
        prev = indices[i]

    return gaps


def loss(labels, clusters): 
    '''data is the array of coordinates, labels is the array of true labels, clusters is the array of assigned labels. 
    This algorithm will go through the list of real and assigned labels and compare cluster sizes to determine how accurate the 
    clustering is.
    It returns the array of error percentages as well as the average error'''

    true_breaks = breaks(labels)
    gaps = sizes(true_breaks)

    errors = []
    beginning = 0
    for i in range(len(gaps)): 
        end = true_breaks[i]
        chunk = clusters[beginning:end]
        mode = st.mode(chunk) # the label that appears most often is the label of the overall event (they won't always match between 
                            # the model and real labels), this assumes that the algorithm can at least recognize an event and measures how 
                            # homogeneous the labelling is 
        appearances = np.count_nonzero(chunk == mode[0])
        errors.append(1-(appearances/gaps[i]))
        beginning = true_breaks[i]

    return errors, np.average(errors)

losses, avg = loss(labels_array, clusters)
print('--------')
print(f'Average loss: {avg}')


