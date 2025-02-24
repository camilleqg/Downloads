import numpy as np 
import matplotlib.pyplot as plt 
import torch
import csv
import pandas as pd  
from scipy import stats as st
from sklearn.preprocessing import MinMaxScaler

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
    
    return clusters.cpu().numpy(), centroids.numpy()

device = "cuda" if torch.cuda.is_available() else "cpu" 

def labelmaker(events, sp_density, t_density, noise, filename = None, folder = None): 
    '''creates labels based on my naming convention for different files, keeps it consistent and easy'''
    if folder: 
        folder = folder + '/'

    if filename: 
        datafile = str(filename) 
    else: 
        datafile = str(events) + 'ev_' + str(sp_density) + 'spd_' + str(t_density) + 'td_n' + str(noise)
    

    labelfile = 'labels_' + datafile + '.csv'
    sourcefile = 'sources_' + datafile + '.csv'
    ai_labelfile = datafile + '_results' + '.csv'
    centroidfile = datafile + '_centroids' + '.csv'
    datafile = datafile + '.csv'

    if folder: 
        datafile = folder + datafile 
        centroidfile = folder + centroidfile 
        ai_labelfile = folder + ai_labelfile 
        labelfile = folder + labelfile 
        sourcefile = folder + sourcefile 

    
    return datafile, labelfile, sourcefile, ai_labelfile, centroidfile

def readfiles(datafile, labelfile, sourcefile): 
    '''data reader and simplifier for files that haven't been passed through the algorithm'''
    columns = ['x[px]', 'y[px]', 't[s]']
    
    dataread = pd.read_csv(datafile) 
    data = np.array(dataread[columns])
    
    labelread = pd.read_csv(labelfile)
    labels = np.array(labelread['labels'])

    sourceread = pd.read_csv(sourcefile) 
    sources = np.array(sourceread[columns])

    return data, labels, sources

def readai(datafile, labelfile, sourcefile, ai_labelfile, centroidfile):
    '''file reader and data simplifier for data thats been through the algorithm'''
    columns = ['x[px]', 'y[px]', 't[s]']
    
    dataread = pd.read_csv(datafile) 
    data = np.array(dataread[columns])
    
    labelread = pd.read_csv(labelfile)
    labels = np.array(labelread['labels'])

    sourceread = pd.read_csv(sourcefile) 
    sources = np.array(sourceread[columns])

    ai_labelsread = pd.read_csv(ai_labelfile)
    ai_labels = np.array(ai_labelsread['labels'])

    centroidread = pd.read_csv(centroidfile) 
    centroids = np.array(centroidread[columns])

    return data, labels, sources, ai_labels, centroids

events = None 
density = None 
noise = None 
filename = None 
folder = None 

datafile, labelfile, sourcefile, ai_labelfile, centroidfile = labelmaker(events, density, noise, filename, folder)
# read out the files
data, labels, sources = readfiles(datafile, labelfile, sourcefile)
# establish feature range and transform data, can comment this out to turn off scaling, naming is the same
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)
sources = scaler.fit_transform(sources)
# turn data into a tensor 
data_tensor = torch.Tensor(data)


# 30 is average number of photons per event
k = 500
max_iters = 500
ai_labels, centroids = kmeans_gpu(data_tensor, k, max_iters)

with open(ai_labelfile, mode = 'w', newline='') as wfile: 
    writer = csv.writer(wfile)
    writer.writerow(['labels'])
    for item in ai_labels: 
                writer.writerow([item])

columns = ['x[px]', 'y[px]', 't[s]']
with open(centroidfile, mode = 'w', newline = '') as wfile: 
    writer = csv.writer(wfile)
    writer.writerow(columns)
    writer.writerows(centroids)

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



