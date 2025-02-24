import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import csv 
from scipy.stats import gaussian_kde
from collections import Counter 
import argparse

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

def readai(datafile, labelfile, sourcefile, ai_labelfile, clusterfile):
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

def mistakes(labels, ai_labels, data): 
    '''sorts out misidentified photons and outputs the correct, incorrect, and correct labels to be used later in the misID function'''
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

def translate(axes): 
    coords = ['x','y','time']

    return coords[axes[0]], coords[axes[1]]

def mono(data, axes, title=None, events=None, sp_density=None, t_density = None, noise=None, figname=None, savefolder = None):
    '''Monochrome simple scatter plot (all blue)'''
    indep, dep = translate(axes)
    if title: 
        title = str(title)
    else: 
        title = f'Monochrome scatterplot of {events} events at {sp_density} spatial density, {t_density} temporal density and {noise} noise photons, {dep} vs {indep}'

    if figname: 
        figname = str(figname)
    else: 
        figname = f'mono_{indep}_vs_{dep}_{events}ev_{sp_density}spd_{t_density}td_n{noise}'
    
    if savefolder: 
        figname = str(savefolder) + '/' + figname 

    plt.figure()
    plt.tight_layout()
    plt.scatter(data[:,axes[0]], data[:,axes[1]])
    plt.title(title)
    plt.ylabel(f'{dep} coordinates')
    plt.xlabel(f'{indep} coordinates')
    plt.savefig(figname)

    print(f"Figure saved under {figname}")

def galaxy(data, axes, title=None, events=None, sp_density=None, t_density = None, noise=None, figname=None, savefolder = None):
    '''Black background 2d scatter plot'''
    indep, dep = translate(axes)
    if title: 
        title = str(title)
    else: 
        title = f'Galaxy map of {events} events at {sp_density} density, {t_density} temporal density, and {noise} noise photons, {dep} vs {indep}'

    if figname: 
        figname = str(figname) + 'png'
    else: 
        figname = f'galaxy_{indep}_vs_{dep}_{events}ev_{sp_density}spd_{t_density}td_n{noise}.png'
    
    if savefolder: 
        figname = str(savefolder) + '/' + figname

    plt.figure()
    plt.tight_layout()
    plt.hist2d(data[:,axes[0]], data[:,axes[1]], bins = (100,100), cmap='hot')
    plt.colorbar()
    plt.title(title)
    plt.ylabel(f'{dep} coordinates')
    plt.xlabel(f'{indep} coordinates')
    plt.savefig(figname)

def colourcoded(data, ai_labels, axes, title=None, events=None, sp_density = None, t_density = None, noise=None, figname=None, savefolder = None):
    '''colour coded scatter plot based on k-means assigned labels '''
    indep, dep = translate(axes)
    if title: 
        title = str(title)
    else: 
        title = f'K-means colour coded scatter plot of {events} events at {sp_density} spatial density, {t_density} temporal density, and {noise} noise photons, {dep} vs {indep}'
    if figname: 
        figname = str(figname) + 'png'
    else: 
        figname = f'colourcode_{indep}_vs_{dep}_{events}ev_{sp_density}spd_{t_density}td_n{noise}.png'
    
    if savefolder: 
        figname = str(savefolder) + '/' + figname
    
    plt.figure()
    plt.tight_layout()
    plt.scatter(data[:,axes[0]], data[:,axes[1]], c = ai_labels, s = 20, cmap = 'tab20')
    plt.title(title)
    plt.ylabel(f'{dep} coordinates')
    plt.xlabel(f'{indep} coordinates')
    plt.savefig(figname)

def centroidVsource(data, ai_labels, sources, clusters, axes, title=None, events=None, sp_density = None, t_density = None, noise=None, figname=None, savefolder = None):
    '''Just like colourcoded but with the true sources and ai cluster centroids compared'''
    indep, dep = translate(axes)
    if title: 
        title = str(title)
    else: 
        title = f'K-means colour coded scatter plot with true sources and cluster centroids of {events} events at {sp_density} spatial density, {t_density} temporal density, and {noise} noise photons, {dep} vs {indep}'
    if figname: 
        figname = str(figname) + 'png'
    else: 
        figname = f'centroidVsource_{indep}_vs_{dep}_{events}ev_{sp_density}spd_{t_density}td_n{noise}.png'
    
    if savefolder: 
        figname = str(savefolder) + '/' + figname

    plt.figure()
    plt.scatter(data[:,axes[0]], data[:,axes[1]], c = ai_labels, s = 20, cmap = 'tab20')
    plt.scatter(sources[:,axes[0]], sources[:,axes[1]], c = 'black', s = 35, marker='x', label = "True Sources")
    plt.scatter(clusters[:,axes[0]], clusters[:,axes[1]], c = 'black', s = 35, marker = 'o', label = 'AI Cluster Centroids')
    plt.title(title)
    plt.ylabel(f'{dep} coordinates')
    plt.xlabel(f'{indep} coordinates')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figname)

def threeD(data, ai_labels, title = None, events = None, sp_density = None, t_density = None, noise = None, figname = None, savefolder = None):
    '''Will produce a 3D scatter plot with colour codes by AI labels'''
    if title: 
        title = str(title)
    else: 
        title = f'K-means colour coded 3D scatter plot of {events} events at {sp_density} spatial density, {t_density} temporal density and {noise} noise photons'
    if figname: 
        figname = str(figname) + 'png'
    else: 
        figname = f'3Dscatter_{events}ev_{sp_density}spd_{t_density}td_n{noise}.png'
    
    if savefolder: 
        figname = str(savefolder) + '/' + figname

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(data[:,2], data[:,0], data[:,1], s = 10, c = ai_labels)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X coordinates')
    ax.set_zlabel('Y coordinates')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(figname)

def misID(data, axes, ai_labels, labels, title = None, events = None, sp_density = None, t_density = None, noise = None, figname = None, savefolder = None):
    '''K-means colour coded scatter plot with misidentified photons coloured in red. Combined or split event photons are not included.'''
    indep, dep = translate(axes)
    if title: 
        title = str(title)
    else: 
        title = f'Misidentified photons (red) of {events} events at {sp_density} spatial density, {t_density} temporal density and {noise} noise photons, {dep} vs {indep}'
    if figname: 
        figname = str(figname) + 'png'
    else: 
        figname = f'misID_{indep}_vs_{dep}_{events}ev_{sp_density}spd_{t_density}td_n{noise}.png'
    
    if savefolder: 
        figname = str(savefolder) + '/' + figname
    
    correct, incorrect, correct_labels = mistakes(labels, ai_labels, data)
    plt.figure()
    plt.scatter(correct[:,axes[0]], correct[:,axes[1]], c = correct_labels, cmap = 'tab20', s = 20)
    plt.scatter(incorrect[:,axes[0]], incorrect[:,axes[1]], c = 'red', s = 20)
    plt.title(title)
    plt.ylabel(f'{dep} coordinates')
    plt.xlabel(f'{indep} coordinates')
    plt.tight_layout()
    plt.savefig(figname)

def plot(plottype, axes, folder = None, filename = None, title = None, events = None, sp_density = None, t_density = None, noise = None, savefolder=None, figname=None):
    '''This function will plot the data in one of the following formats: 
    0. mono: monochrome scatter plot, no other indicators
    1. galaxy: black background with density based colour gradient
    2. colourcoded (ccd): scatter plot with points coloured based on the ai label
    3. centroidVsource (cvs): scatter plot that shows the cluster centroids vs the source coordinates, colour coded like option 2
    4. 3D: 3D colour coded plot 
    5. misidentified (misID): highlights misidentified photons in red 
    '''

    types = ['mono', 'galaxy', 'colourcoded', 'centroidVsource', '3D', 'misidentified']

    # generate datafile name 
    datafile, labelfile, sourcefile, ai_labelfile, clusterfile = labelmaker(events, sp_density, t_density, noise, filename, folder) 
    # should only read the data that it needs 
    val = types.index(str(plottype))
    if val > 1: 
        data, labels, sources, ai_labels, clusters = readai(datafile, labelfile, sourcefile, ai_labelfile, clusterfile)
    else: 
        data, labels, sources = readfiles(datafile, labelfile, sourcefile)
    
    if str(plottype) == 'mono': 
        mono(data, axes, title, events, sp_density, t_density, noise, figname, savefolder)
    elif str(plottype) == 'galaxy': 
        galaxy(data, axes, title, events, sp_density, t_density, noise, figname, savefolder)
    elif str(plottype) == 'ccd': 
        colourcoded(data, ai_labels, axes, title, events, sp_density, t_density, noise, figname, savefolder)
    elif str(plottype) == 'cvs': 
        centroidVsource(data, ai_labels, sources, clusters, axes, title=None, events=None, sp_density=None, t_density=None, noise=None, figname=None, savefolder = None)
    elif str(plottype) == '3D': 
        threeD(data, ai_labels, title = None, events = None, sp_density=None, t_density=None, noise = None, figname = None, savefolder = None)
    elif str(plottype) == 'misID':
        misID(data, axes, ai_labels, labels, title = None, events = None, sp_density=None, t_density=None, noise = None, figname = None, savefolder = None)
    else: 
        print("Plot type not recognized. Please check and try again. Options are: [mono, galaxy, ccd, cvs, 3D, misID]")

parser = argparse.ArgumentParser(description="Plotting functions!")

subparsers = parser.add_subparsers(dest="command", help = "Available commands")

plt_parser = subparsers.add_parser("plot", help = "Plot data under several formats")

plot('mono', [1,2], events = 10, density = '1', noise = 0, figname = 'test1')

plt_parser.add_argument("-p", "--plottype", type = str, required = True, help = "Type of plot")
plt_parser.add_argument("-a", "--axes", type = list, required = True, help = "axes to display in the plot given in [,] format. for 3D plot put anything, the default will be used for this")
plt_parser.add_argument("-fd", "--folder", type = str, default = None, help = "folder in which the data is saved, needed if the default filename format is being read.")
plt_parser.add_argument("-fn", "--filename", type = str, default = None, help = "name of the datafile, needed if the default filename format is being read.")
plt_parser.add_argument("-t", "--title", type = str, default = None, help = "Optional custom title to give the plot")
plt_parser.add_argument("-e", "--events", type = int, default = None, help = "Number of events in the datafile, needed if default filename format is being read.")
plt_parser.add_argument("-spd", "--spacedensity", type = float, default = None, help = "Spatial density of photons in datafile. Needed if default filename is being read.")
plt_parser.add_argument("-td", "--timedensity", type = float, default = None, help = "")
plt_parser.add_argument("-n", "--noise", type = int, default = None, help = "Number of noise photons included in the datafile, needed if default filename is being read")
plt_parser.add_argument("-sf", "--savefolder", type = str, default = None, help = "Optional folder in which to save the figure")
plt_parser.add_argument("-fig", "--figname", type = str, default = None, help = "Optional custom name under which to save the figure")

args = parser.parse_args()

# call the function based on subcommand
if args.command == "plot": 
    plot(plottype = args.plottype, axes=args.axes, folder = args.folder, filename=args.filename, title=args.title, events=args.events, density=args.density, noise=args.noise, savefolder=args.savefolder, figname=args.figname)
else:    
    parser.print_help()
