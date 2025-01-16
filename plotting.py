import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import csv 
from scipy.stats import gaussian_kde
from collections import Counter 

def labelmaker(events, density, noise, folder = None): 
    folder = folder + '/'

    datafile = str(events) + 'ev_' + str(density) + 'dense_n' + str(noise) + '.csv'
    clusterfile = str(events) + 'ev_' + str(density) + 'dense_n' + str(noise) + '_centroids' + '.csv'
    ai_labelfile = str(events) + 'ev_' + str(density) + 'dense_n' + str(noise) + '_results' + '.csv'
    labelfile = 'labels_' + str(events) + 'ev_' + str(density) + 'dense_n' + str(noise) + '.csv'
    sourcefile = 'sources_' + str(events) + 'ev_' + str(density) + 'dense_n' + str(noise) + '.csv'

    if folder: 
        datafile = str(folder)+ datafile
        clusterfile = str(folder) + clusterfile 
        ai_labelfile = str(folder) + ai_labelfile
        labelfile = str(folder) + labelfile 
        sourcefile = str(folder) + sourcefile 
    
    return datafile, labelfile, sourcefile, ai_labelfile, clusterfile

def readfiles(events, density, noise, folder = None): 
    datafile, labelfile, sourcefile, ai_labelfile, clusterfile = labelmaker(events, density, noise, folder)
    columns = ['x[px]', 'y[px]', 't[s]']
    
    dataread = pd.read_csv(datafile) 
    data = np.array(dataread[columns])
    
    labelread = pd.read_csv(labelfile)
    labels = np.array(labelread['labels'])

    sourceread = pd.read_csv(sourcefile) 
    sources = np.array(sourceread[columns])

    ai_labelsread = pd.read_csv(ai_labelfile)
    ai_labels = np.array(ai_labelsread['labels'])

    clusterread = pd.read_csv(clusterfile) 
    clusters = np.array(clusterread[columns])

    return data, labels, sources, ai_labels, clusters

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

def modded_mode(array): 
    '''This function will return event labels, its a modification on a normal mode function where the label has to appear at least 33% of the time to be a label in that chunk.'''
    n = len(array)
    if n == 0: 
        return []
    
    threshold = n/3 
    counts = Counter(array)

    result = [key for key, count in counts.items() if count > threshold]
    return result 

data, labels, sources, ai_labels, clusters = readfiles(10, '.25', 0, '10events')

def mistakes(labels, ai_labels, data): 
    correct = []
    incorrect = []
    correct_labels = []
    true_breaks = breaks(labels)

    start = 0 
    for end in true_breaks: 
        chunk = ai_labels[start:end] #isolate the corresponding ai labels from each true event 
        ai_lbls = modded_mode(chunk) #find the mode(s) for these chunks
        for i in range(len(chunk)):
            if chunk[i] in ai_lbls:
                correct.append(data[start+i])
                correct_labels.append(chunk[i])
            else: 
                incorrect.append(data[start+i])


        start = end 

    return np.array(correct), np.array(incorrect), np.array(correct_labels)

maxtime = np.max(data[:,2])
maxy = np.max(data[:,1])
maxx = np.max(data[:,0])

## scatter with colour code based on labels 
plt.figure()
plt.scatter(data[:,2], data[:,1], c = ai_labels, s = 20, cmap='tab20')
plt.scatter(sources[:,2], sources[:,1], c = 'black', s= 35, marker='x')
plt.title('Time vs Y of 10 events .25 dense true sources', wrap =True)
plt.ylabel('Y coordinates (m)')
plt.xlabel('Time (s)')
#plt.xlabel('X coordinates (m)')
plt.savefig('kmeans_v2_10ev_n0_.25dense_time.png')

# scatter comparison of true sources and cluster centroids 
plt.figure()
plt.scatter(sources[:,2], sources[:,1], c = 'black', s= 35, marker='x')
plt.scatter(clusters[:,2], clusters[:,1], c = 'blue', s = 35, marker = 'o')
plt.title('Sources vs Cluster centroids for 10 events .25 dense', wrap = True)
plt.ylabel('Y coordinates (m)')
plt.xlabel('Time (s)')
plt.savefig('kmeans_v2_sourcesvscentroids_n0_.25dense_time.png')


# scatter 
# plt.scatter(points[:,2], points[:,1])
# #plt.scatter(sourcepoints[:,2], sourcepoints[:,1], c='r')
# plt.title('Y vs time view of 10 events at .5 dense')
# plt.ylabel('Y coordinates (m)')
# plt.xlabel('Time (s)')
# plt.savefig('10ev_.5dense_n0.png')


# fix one of the next two plot blocks to produce what im looking for 
# histogram 
# plt.hist2d(points[:,0], points[:,1], cmap = 'hot', bins = [255, 255])
# plt.title('Y vs x view of 20 events')
# plt.colorbar()
# plt.ylabel('Y coordinates (m)')
# plt.xlabel('X coordinates (m)')
# plt.savefig('20ev_realt_hist.png')

# black background? -> density c
# plt.figure() 
# # cbar = plt.colorbar(hist[3], pad=0.01)
# # cbar.set_label('Density')
# plt.hist2d(data[:,0], data[:,1], bins = (100,100), cmap='hot')
# plt.colorbar()
# plt.xlabel("X coordinates")
# plt.ylabel("Y coordinates")
# plt.title("10 events, x vs y view")
# plt.tight_layout()
# plt.savefig('10evtest.png')



# mapmap = points[:,1:3]
# plt.imshow(points[:, 1:3])
# plt.title('Heatmap y vs time view of 10 events')
# plt.ylabel('Y coordinates (m)')
# plt.xlabel('Time (s)')
# plt.savefig('20ev_realt_heat.png')
# #print(mapmap.shape)

# 3D plot 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:,2], points[:,0], points[:,1], s=10)

# ax.set_xlabel('Time (s)')
# ax.set_ylabel('X coordinates')
# ax.set_zlabel('Y coordinates')
# ax.set_title('3D plot of 10 events .25 density')
# plt.savefig('3D_10ev_.25dense_n0.png')


# misidentified photons 
# correct, incorrect, correct_labels = mistakes(labels, ai_labels, data)

# plt.figure()
# plt.scatter(correct[:,0], correct[:,1], c = correct_labels, cmap = 'tab20', s = 10)
# plt.scatter(incorrect[:,0], incorrect[:,1], c = 'red', s = 10)
# plt.title('X Y view of (in) correctly identified photons, 10 events, .5 density')
# plt.ylabel('Y coordinates')
# plt.xlabel('X coordinates')
# plt.savefig('misID_xy_10ev_.5dense.png')

# plt.figure()
# plt.scatter(correct[:,2], correct[:,1], c = correct_labels, cmap = 'tab10', s = 10)
# plt.scatter(incorrect[:,2], incorrect[:,1], c = 'red', s = 10)
# plt.title('Time Y view of (in) correctly identified photons, 10 events, .5 density')
# plt.ylabel('Y coordinates')
# plt.xlabel('Time coordinates')
# plt.savefig('misID_ty_10ev_.5dense.png')

# plt.figure()
# plt.scatter(correct[:,2], correct[:,0], c = correct_labels, cmap = 'tab20', s = 10)
# plt.scatter(incorrect[:,2], incorrect[:,0], c = 'red', s = 10)
# plt.title('Time X view of (in) correctly identified photons, 10 events, .5 density')
# plt.ylabel('X coordinates')
# plt.xlabel('Time coordinates')
# plt.savefig('misID_tx_10ev_.5dense.png')
