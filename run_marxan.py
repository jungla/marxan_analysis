import os
import subprocess

label = 'TUMCA'

# create input file

#for BLM in ['0','0.001','0.01','0.1','1','10']:
for BLM in ['0.01']:
 print(BLM)
 input = '''
 General Parameters
 BLM '''+BLM+'''
 PROP 0.5
 RANDSEED -1
 NUMREPS 500
 
 Annealing Parameters
 NUMITNS 1000000
 STARTTEMP -1
 NUMTEMP 10000
 
 Cost Threshold
 COSTTHRESH  0.00000000000000E+0000
 THRESHPEN1  1.40000000000000E+0001
 THRESHPEN2  1.00000000000000E+0000
 
 Input Files
 INPUTDIR input
 PUNAME pu.dat
 SPECNAME spec.dat
 PUVSPRNAME puvspr.dat
 BOUNDNAME bound.dat
 
 Save Files
 SCENNAME output
 SAVERUN 3
 SAVEBEST 3
 SAVESUMMARY 3
 SAVESCEN 3
 SAVETARGMET 3
 SAVESUMSOLN 3
 SAVEPENALTY 3
 SAVELOG 2
 OUTPUTDIR output
 
 Program control.
 RUNMODE 1
 MISSLEVEL 1
 ITIMPTYPE 0
 HEURTYPE -1
 CLUMPTYPE 0
 VERBOSITY 3
 
 SAVESOLUTIONSMATRIX 3
 '''

 print(input)
 
 fname = '/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Research/Analysis/Marxan/MarxanData/'+label+'/input.dat'
 f = open(fname, 'wb')
 f.write(input.encode('utf-8'))
 f.close()
 
 # run marxan
 
 subprocess.call('./marxan', cwd='/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Research/Analysis/Marxan/MarxanData/'+label+'/', shell=True)

 subprocess.call(['mkdir', '-p', 'BLM'+BLM], cwd='/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Research/Analysis/Marxan/MarxanData/'+label+'/')
 subprocess.call(['cp', '-fr', 'output','BLM'+BLM], cwd='/Users/jeanmensa/My Drive (jmensa@wcs.org)/WCS Tanzania Marine Program/Research/Analysis/Marxan/MarxanData/'+label+'/')
