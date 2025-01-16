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

def loss(true_labels, network_labels):
    all_modes =[]
    event_splits = 0 
    all_misidentified = [] # array for all fractions of misidentified events 

    break_indices = breaks(true_labels)
    gaps = sizes(break_indices) # sizes of each event 

    beginning = 0 
    for i in range(len(gaps)):
        misidentified=0
        end = break_indices[i]
        chunk = network_labels[beginning:end]
        chunk_modes = modded_mode(chunk) # find the modes 
        all_modes.extend(chunk_modes) # add to master list of modes 
        if len(chunk_modes) > 1: 
            event_splits += 1  # add the number of splits to the total amount of splits 

        chunk_modes_set = chunk_modes # change the list of modes into a set to increase efficiency
        misidentified = sum(1 for item in chunk if item not in chunk_modes_set) # counts the number of photons not included in the main event labels (modes)
        err_fraction = misidentified/gaps[i] # calculates the fraction of misidentified over number of photons in the event 
        all_misidentified.append(err_fraction) # adds the misidentification error to array 
        beginning = break_indices[i]
    
    avg_misidentified = np.average(all_misidentified)
    counter = Counter(all_modes)
    unfiltered_counts = dict(counter) # look at how many times each mode or "label" shows up in this list 
    mode_modes = {item: count for item, count in counter.items() if count > 1} # if it's more than once then an event has been combined 
    combo_frac = len(mode_modes.values())/len(unfiltered_counts.values())
    avg_ev_in_combo = sum(mode_modes.values())/len(mode_modes.values())

    print(f"The fraction of misidentified photons in each event is: {all_misidentified}")
    print(f"the average fraction of misidentified photons is: {avg_misidentified}") 
    print(f"The total number of event splits is {event_splits}")
    print(f"The full list of events that were combined is: {mode_modes}, with a fraction of {combo_frac} events being combined. The average number of events involved in a combination is: {avg_ev_in_combo}")
    # the mode modes here refer to the labels given to the events by the ai, not the true labels (for reference) 
    return event_splits, all_misidentified, avg_misidentified, mode_modes, combo_frac, avg_ev_in_combo
    
    
    # print(unfiltered_counts)
    # print(mode_modes)
    # print(unfiltered_counts.values())
    # print(avg_combination)


    # print(all_misidentified)
    # print(gaps)

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

loss(labels, ai_labels)