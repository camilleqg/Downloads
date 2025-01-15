# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:07:05 2024

@author: camil
"""

import numpy as np 
import matplotlib.pyplot as plt
import textwrap
import argparse 
import json
import csv
import pandas as pd
import random 
import sklearn
from sklearn.utils import shuffle
import multiprocessing as mp
from multiprocessing import Pool, cpu_count 

# buncha constants 

event_density = 487 # events per square cm in one second
detector_sidelength = 1e-2 # in metres 
#time_window = 0.3

# import json file stuff for num_events 
def load_params(json_filename):
    '''This just loads parameters from the json file'''
    with open(json_filename, 'r') as f: 
        params = json.load(f)
    return params

# picking total number of photons 

def num_events(photons, a0 = 3426.89, b0 = -3476.32, c0 = 0.10182, sigma = 40.296, mu = 24.3823, file = None):
    '''outputs the probability of events for a given number of photons based on the best fit of the histogram'''
    if file:
        params = load_params(file) 
        a0 = params.get('a0', 3426.89)
        b0 = params.get('b0', -3476.32)
        c0 = params.get('c0', 0.10182)
        sigma = params.get('sigma', 40.296)         
        mu = params.get('mu', 24.3823)
    return 1/6*(a0*np.exp(-(photons-mu)**2 / (2*sigma**2)) + b0*np.exp(-c0*photons)) 

def total_photons(scale, file = None):
    '''guesses the total number of photons out of possible values as well as a number of events.
    from there it checks if these points appear together on the histogram. the probability of coincidence
    matches the histogram distribution. once it finds a number it likes it spits it out'''
    photonspace = np.linspace(0,255,500) # this and next two lines samples function and finds an upper limit to number of events
    eventspace = num_events(photonspace) 
    max_events = max(eventspace)
    
    min_photons = 1 # upper and lower limit of number of photons 
    max_photons = 255
    
    # evaluates the num_events function to see if the random numbers match the distribution
    num_photons_final = 0 
    while num_photons_final == 0: 
        # generates a random number of photons and random number of events based on the limits
        guess_photons = random.uniform(min_photons, max_photons)
        guess_events = random.uniform(1, max_events)
        
        if guess_events <= num_events(guess_photons, file = file): 
            num_photons_final = guess_photons
    return round(scale*num_photons_final)

# generate coordinates for one photon 
def generate_coords(mu_x, mu_y, sigma2=0.00021233045007200478): 
    ''' This function will generate a set of coordinates for x and y, based on the centre of the event (input). It will only generate one at a time
    no need for custom functions here because numpy already has a sampler for gaussian distribution'''
    num_photons = 1
    x_coord = np.random.normal(loc = mu_x, scale = sigma2, size = num_photons)
    y_coord = np.random.normal(loc = mu_y, scale = sigma2, size = num_photons)
    return x_coord[0], y_coord[0]

# generate time coordinate for one photon


def decayfit(t, a1 = 57782.4, t1 = 0.000653566, a2 = 7473.2, t2 = 0.016498, a3 = 1.28054e6, t3 = 2.87915e-05, a4 = 455714, t4 = 0.000119424, file = None): 
    ''' This function takes the time input and outputs f(x) of the best fit line of the decay'''
    if file: 
        params = load_params(file)
        a1 = params.get('a0', 57782.4) 
        t1 = params.get('t0', 0.000653566)
        a2 = params.get('a1', 7473.2)
        t2 = params.get('t1', 0.016498)
        a3 = params.get('a2', 1.28054e6)
        t3 = params.get('t2', 2.87915e-05)
        a4 = params.get('a3', 455714)
        t4 = params.get('t3', 0.000119424)
    return a1*np.exp(-t/t1) + a2*np.exp(-t/t2) + a3*np.exp(-t/t3) + a4*np.exp(-t/t4)

def generate_time(file):
    ''' Just like the photon number generator, will guess a time and number of photons, if these points
    coincide on the integral of the best fit then they are accepted and returned '''
    # event box limits 
    time_min = 0
    time_max = 0.1 # max time for a single event 
    
    # max number of photons at a time 
    max_photons = decayfit(t = 0, file = file)
    
    # loop through until we get an accepted value 
    time_final = 0 
    while time_final == 0: 
        time_guess = np.random.uniform(time_min, time_max)
        photons_guess = np.random.uniform(0, max_photons)
        if photons_guess <= decayfit(time_guess, file = file):
            time_final = time_guess 
            
    return time_final

def scramble(x, y, state=0):
    new_x, new_y = shuffle(x, y, random_state=state)

    return new_x, new_y  


def sim(events, noise, total_events, mix=False, verbose = False, eventscale=1, spacesigma=0.00021233045007200478, start_time=-1, start_x=-1, start_y=-1, dataSaveID = None, file=None):
    # if theres a filename, change the relevant parameters 
    # pid= mp.current_process().pid
    # rng = np.random.default_rng(seed=pid)

    time_window = total_events/event_density # in seconds

    if file: 
        params = load_params(file)
        verbose = params.get('verbose', False)
        eventscale = params.get('eventscale', 1) 
        spacesigma = params.get('spacesigma', 0.00021233045007200478)
        start_time = params.get('start_time', -1)
        start_x = params.get('start_x', -1)
        start_y = params.get('start_y', -1)

    
    # initialize master lists
    master_data = []
    master_labels = []
    master_sources= []
    
    for i in range(events):
        
        # label = rng.integers(1, 10000)
        # num_photons = total_photons(eventscale, file=file)
        # event_labels = [label]*num_photons
        # if start_time == -1: 
        #     start_time = rng.uniform(0, time_window)
        # if start_x == -1: 
        #     start_x = rng.uniform(0, detector_sidelength)
        # if start_y == -1: 
        #     start_y = rng.uniform(0, detector_sidelength)
        
        label = random.randint(1, 10000)
        num_photons = total_photons(eventscale, file=file)
        event_labels = [label]*num_photons
        if start_time == -1: 
            start_time = random.uniform(0, time_window)
        if start_x == -1: 
            start_x = random.uniform(0, detector_sidelength)
        if start_y == -1: 
            start_y = random.uniform(0, detector_sidelength)

        
        for j in range(num_photons): 
            coords = [0]*3
            coords[0] = (generate_coords(start_x, start_y, sigma2=spacesigma)[0])
            coords[1] = (generate_coords(start_x, start_y, sigma2=spacesigma)[1])
            coords[2] = (generate_time(file = file) + start_time)
            master_data.append(coords)
        
        master_labels.append(event_labels)
        master_sources.append([start_x, start_y, start_time])
    
    # NORMALIZATION HERE 
    # for i in range(noise): 
    #     coords = [0]*3 
    #     coords[0] = rng.uniform(0, 1)
    #     coords[1] = rng.uniform(0, 1) 
    #     coords[2] = rng.uniform(0, 1)
    #     master_data.append(coords)
    
    for i in range(noise): 
        coords = [0]*3
        coords[0] = random.uniform(0,detector_sidelength)
        coords[1] = random.uniform(0,detector_sidelength)
        coords[2] = random.uniform(0,time_window)
        master_data.append(coords)

    master_labels.append([0]*noise)
    
    data = np.array(master_data) 
    labels = np.array([item for sublist in master_labels for item in sublist])
    sources = np.array(master_sources)
    
    if mix==True: 
        data, labels = scramble(data, labels)
    
    if verbose: 
        for sublist in master_data: 
            print(sublist)
        print('\n--------\n')
        print(labels)
        print('\n--------\n')
        for sublist in master_sources: 
            print(sublist)
    # else: 
    #     print("output suppressed")
    
    return data, labels, sources 

def parallel_sim(events, num_cores, noise, mix=False, verbose = False, eventscale=1, spacesigma=0.00021233045007200478, start_time=-1, start_x=-1, start_y=-1, dataSaveID = None, file=None):
    assert 0 < num_cores < mp.cpu_count()
    e_per_core = events//num_cores
    n_per_core = noise//num_cores 
    core_args = [
        (e_per_core, n_per_core, mix, verbose, eventscale, spacesigma, start_time, start_x, start_y, dataSaveID, file)
        for i in range(num_cores)]
    
    results = None 
    with mp.Pool(num_cores) as pool: 
        results = pool.starmap(sim, core_args)
    
    event_data = []
    labels = []
    sources = []
    
    for core_data, core_labels, core_sources in results: 
        event_data.extend(core_data)
        labels.extend(core_labels)
        sources.extend(core_sources)

    
    if dataSaveID: 
        columns = ['x[px]', 'y[px]', 't[s]']
        truthsaveID = 'labels_' + dataSaveID
        sourcesaveID = 'sources_' + dataSaveID
        with open(dataSaveID, mode = 'w', newline='') as wfile: 
            writer = csv.writer(wfile)
            writer.writerow(columns)
            writer.writerows(event_data)
        with  open(truthsaveID, mode = 'w', newline = '') as wfile: 
            writer = csv.writer(wfile)
            writer.writerow(['labels'])
            for item in labels: 
                writer.writerow([item])
        with open(sourcesaveID, mode = 'w', newline = '') as wfile: 
            writer = csv.writer(wfile)
            writer.writerow(columns)
            writer.writerows(sources)
        
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


def new_parallel_sim(events, num_cores, noise, density ='1', folder = None, mix=False, verbose = False, eventscale=1, spacesigma=0.00021233045007200478, start_time=-1, start_x=-1, start_y=-1, dataSaveID = True, file=None):
    
    datafile, labelfile, sourcefile, ai_labelfile, clusterfile = labelmaker(events, density, noise, folder)
    
    assert 0 < num_cores < mp.cpu_count()
    runs_per_core = events//num_cores
    n_per_core = noise//num_cores//events
    core_args = [
        (1, n_per_core, events, mix, verbose, eventscale, spacesigma, start_time, start_x, start_y, dataSaveID, file)
        for i in range(num_cores)]
    
 
    event_data = []
    labels = []
    sources = []

    with mp.Pool(num_cores) as pool: 
        for i in range(runs_per_core): 
            results = None
            results = pool.starmap(sim, core_args)
            for core_data, core_labels, core_sources in results: 
                event_data.extend(core_data)
                labels.extend(core_labels)
                sources.extend(core_sources)
        
    if dataSaveID: 
        columns = ['x[px]', 'y[px]', 't[s]']
        with open(datafile, mode = 'w', newline='') as wfile: 
            writer = csv.writer(wfile)
            writer.writerow(columns)
            writer.writerows(event_data)
        with open(labelfile, mode = 'w', newline = '') as wfile: 
            writer = csv.writer(wfile)
            writer.writerow(['labels'])
            for item in labels: 
                writer.writerow([item])
        with open(sourcefile, mode = 'w', newline = '') as wfile: 
            writer = csv.writer(wfile)
            writer.writerow(columns)
            writer.writerows(sources)



def data_reader(filename): 
    labelname = 'labels_' + filename
    with open(filename, 'r') as file: 
        reader = csv.reader(file)
        for row in reader: 
            print(row)

    with open(labelname, 'r') as lfile: 
        reader = csv.reader(lfile)
        for row in reader: 
            print(row)

parser = argparse.ArgumentParser(description="Neutron event simulator")

# creating sub parsers for each command (functions) 
# each of these functions get their own arguments 

subparsers = parser.add_subparsers(dest="command", help = "Available commands")

#def sim(events, noise, scramble, verbose = False, eventscale=1, noisescale=1, spacesigma=0.00021233045007200478, start_time=-1, start_x=-1, start_y=-1, dataSaveID = None, file=None):

sim_parser = subparsers.add_parser("simulate", help = "Simulate neutron events, output results into terminal")
sim_parser.add_argument("-e", "--events", type = int, required=True, help="Number of neutron events to simulate")
sim_parser.add_argument("-c", "--cores", type=int, required=True, help="Number of cpu cores to be used")
sim_parser.add_argument("-n", "--noise", type = int, required = True, help="Number of noise photons to include in the data")
sim_parser.add_argument("-d", "--density", type = str, default = '1', help="Density of events in time and space, from 0-1, needs to be a string with no leading zeroes")
sim_parser.add_argument("-f", "--folder", type =str, default = None, help = "Folder in which to save the datafiles.")
sim_parser.add_argument("-m", "--mix", type = bool, default=False, help="True for mixing the data, False for leaving it in order")
sim_parser.add_argument("-v", "--verbose", type = str, default = False, help="Showing the photon data in terminal or not")
sim_parser.add_argument("-es", "--eventscale", type = float, default = 1, help = "Scaling coefficient for number of photons per event. Default value: 1")
sim_parser.add_argument("-s", "--spacesigma", type = float, default = 0.00021233045007200478, help = "Sigma of the Gaussian distribution of photons in x and y. Default value: 0.00021233045007200478")
sim_parser.add_argument("-st", "--starttime", type = float, default = -1, help = "Start time for all events. Default value is randomly generated.")
sim_parser.add_argument("-sx", "--startx", type = float, default = -1, help = "Starting x coordinate for all events. Default value is randomly generated.")
sim_parser.add_argument("-sy", "--starty", type = float, default = -1, help = "Starting y coordinate for all events. Default value is randomly generated.")
sim_parser.add_argument("-df", "--datafile", type = None, default = True, help = "Text file name for saving data. Truth file will have the same name but prefixed with truth_")
sim_parser.add_argument("-jf", "--file", type = None, default = None, help = "Name of json file containing all optional arguments as well as parameters for photon decay distribution and distribution of number of photons per event.")

read_parser = subparsers.add_parser("read", help="Read and print csv file, only need the name of the data file")
read_parser.add_argument("-id", "--filename", type=str, required=True, help="File name")

args = parser.parse_args()

#events, num_cores, noise, density ='1', folder = None, mix=False, verbose = False, eventscale=1, spacesigma=0.00021233045007200478, start_time=-1, start_x=-1, start_y=-1, dataSaveID = True, file=None

# call the function based on subcommand
if args.command == "simulate": 
    new_parallel_sim(events=args.events, num_cores=args.cores, noise=args.noise, density=args.density, folder=args.folder, mix=args.mix, verbose=args.verbose, eventscale=args.eventscale, spacesigma=args.spacesigma, start_time=args.starttime, start_x=args.startx, start_y=args.starty, dataSaveID=args.datafile, file=args.file)
elif args.command == "read": 
    data_reader(filename=args.filename)
else: 
    parser.print_help()
    
    
    
    
    
    
    