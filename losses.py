import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats as st
from collections import Counter 
import pandas as pd
from statistics import multimode

# this will contain testing and building of three loss functions for the three types of errors in k-means results 
# will also turn it into an algorithm that will do them all in a row with less memory usage

### algorithm 

# go through ground truth list, make a list of the breaks in labels 
# index each section of the resultant label list (each supposed event)
# for each event 
    # identify and save the list of modes 
    # length of list of modes: if it's more than one the event is split -> event splitting 
    # take fraction of the event that isn't labelled as one of the modes -> event misidentification, catch the stragglers
# go through list of all modes and check for repititions -> event combination

def modded_mode(array): 
    '''This function will return event labels, its a modification on a normal mode function where the label has to appear at least 33% of the time to be a label in that chunk.'''
    n = len(array)
    if n == 0: 
        return []
    
    threshold = n/3 
    counts = Counter(array)

    result = [key for key, count in counts.items() if count > threshold]
    return result 

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


def truth_based_loss(true_labels, network_labels):
    # labels are already sorted by true labels predictions 

    # getting break indices and cluster sizes 
    break_indices = breaks(true_labels)
    gaps = sizes(break_indices)
    n_events = len(gaps)

    # initialize variables 
    counter = Counter()
    fractions_misIDs = []
    total_splits, ev_per_split = 0,0 
    total_splits = 0 


    # process each chunk that's separated by truth gaps 
    for start, end, gap in zip([0] + break_indices[:-1], break_indices, gaps):
        chunk = ai_labels[start:end]
        chunk_modes = modded_mode(chunk)

        # update counter
        counter.update(chunk_modes)

        e_in_split = len(chunk_modes) # number of ai events found in the chunk
        if e_in_split > 1: 
            total_splits += 1
            ev_per_split += e_in_split # if there is more than one, increase splits and events in split 
        
        misIDs = sum(1 for item in chunk if item not in chunk_modes)
        fractions_misIDs.append(misIDs/gap)

    
    # Combination Statistics 
    repeat_labels = {k: v for k, v in counter.items() if v>1}
    total_combos = len(repeat_labels) # the amount of combinations is the same as the amount of labels that get repeated through the set 
    ev_per_combo = sum(repeat_labels.values())/total_combos if total_combos else 0 # sum all repeats together and average over number of combinations 
    frac_combos = total_combos / n_events # fraction of events experiencing combination

    # other stats 
    ev_per_split = ev_per_split/total_splits if total_splits else 0
    frac_splits = total_splits/n_events
    avg_misIDs = np.mean(fractions_misIDs)

    # output da resultssss
    print(f"The fraction of splits over all events is {frac_splits}")
    print(f"The average number of events involved in a single split is {ev_per_split}")
    print(f"The fraction of combinations over all events is {frac_combos}")
    print(f"The average number of events involved in a single combo is {ev_per_combo}")
    print(f"The average number of photons misidentified in each event is {avg_misIDs}")

    return frac_splits, ev_per_split, frac_combos, ev_per_combo, avg_misIDs

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


data, labels, sources, ai_labels, clusters = readfiles(10, '.25', 0, '10events')

truth_based_loss(labels, ai_labels)