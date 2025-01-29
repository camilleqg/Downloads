Running the simulation: 
type in the terminal: 
    python [filepath] simulate 

followed by any arguments you want to include:  

-e: number of events to simulate  
-n: number of noise photons to be added to the datafile 
-c: number of cores to use, the default is 1, do not include this argument for now to keep the simulation non-parallel until I can debug  
-d: temporal density of events. This can be ignored and left at the default of 1 
-f: folder in which to save the file - if any  
-m: boolean argument (default False) that determines whether the data is mixed (True) or whether it stays in order so all photons in the same event are following one another (False)   
-v: verbose, whether to output results in the terminal, default is False    
-es: eventscale, float number to scale the number of photons per event up or down, default 1     
-s: sigma value for the Gaussian distribution of photons in space, default is that of the N11 scintillator     
-st: start time for all events, default is randomly generated 
-sx: starting x coordinate for all events, default is randomly generated  
-sy: starting y coordinate for all events, default is randomly generated  
-df: file name for saving the data, if left blank my standard naming convention will be applied. See Glossary file for more info on this.  
-jf: name of an optional json file that the simulation can read that includes any or all of the following arguments:  
  - verbose
  - eventscale
  - sigma value
  - starting time
  - start x
  - start y

*The only required arguments are -e and -n, I usually keep n at 0* 
