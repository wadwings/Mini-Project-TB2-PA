


import os
import random
import string
import time
import numpy as np
import math

###################

#Input
q = 10000 ## EDIT number of feature vectors per histogram
p = 3
howmany = 100 ## EDIT number of histograms per sample
class_time = "m26" ## EDIT this is what you want output to be called, see out_file variable

#EDIT this is the input file of CDR3s to be analysed
filename = "/home/path/to/files/cdr3_file.txt"

#EDIT this is where output file (containing frequency distributions) should go
pathout = "/home/path/to/files/"
code_file = "codewords"
codewords = np.loadtxt(pathout+code_file+'.txt', delimiter=',')
outfile = open(pathout+'results'+class_time+'.txt', "w")

def atchley_factor(x):
    import collections as coll
    m = len(x)
    lookup = [[0.591, 1.302, 0.733, 1.570, 0.146],
              [1.343, 0.465, 0.862, 1.020, 0.255],
              [1.050, 0.302, 3.656, 0.259, 3.242],
              [1.357, 1.453, 1.477, 0.113, 0.837],
              [1.006, 0.590, 1.891, 0.397, 0.412],
              [0.384, 1.652, 1.330, 1.045, 2.064],
              [0.336, 0.417, 1.673, 1.474, 0.078],
              [1.239, 0.547, 2.131, 0.393, 0.816],
              [1.831, 0.561, 0.533, 0.277, 1.648],
              [1.019, 0.987, 1.505, 1.266, 0.912],
              [0.663, 1.524, 2.219, 1.005, 1.212],
              [0.945, 0.828, 1.299, 0.169, 0.933],
              [0.189, 2.081, 1.628, 0.421, 1.392],
              [0.931, 0.179, 3.005, 0.503, 1.853],
              [1.538, 0.055, 1.502, 0.440, 2.897],
              [0.228, 1.399, 4.760, 0.670, 2.647],
              [0.032, 0.326, 2.213, 0.908, 1.313],
              [1.337, 0.279, 0.544, 1.242, 1.262],
              [0.595, 0.009, 0.672, 2.128, 0.184],
              [0.260, 0.830, 3.097, 0.838, 1.512]]
    aa = coll.defaultdict(int)
    aa['A'] = 0; aa['C'] = 1; aa['D'] = 2; aa['E'] = 3
    aa['F'] = 4; aa['G'] = 5;aa['H'] = 6;aa['I'] = 7
    aa['K'] = 8;aa['L'] = 9;aa['M'] = 10;aa['N'] = 11
    aa['P'] = 12;aa['Q'] = 13;aa['R'] = 14;aa['S'] = 15
    aa['T'] = 16;aa['V'] = 17;aa['W'] = 18; aa['Y'] = 19

    xsplit = list(x)
    xfactors = [0] * (5 * m)

    for i in range(m):
        for j in range(5):
            xfactors[5 * i + j] = lookup[aa[xsplit[i]]][j]

    return xfactors
count = 0
counter = 0
numvects = 0
t0 = time.time()
histocount = [0] * len(codewords)
print('Mapping Atchley vectors to codewords...')


seqs = []
for line in open(filename, "r"):
    line = line.rstrip("\n")
    seqs.append(line)

while counter < howmany:
    ## Sample q times to generate freq dist over codewords
    while count < q:
        pickseq = random.randint(0, len(seqs) - 1)
        m = len(seqs[pickseq])
        if m > p:
            # start of p mer must be located p steps from end
            picktriplet = random.randint(0, m - p)
            x = seqs[pickseq][picktriplet:picktriplet + p]
            af = str(atchley_factor(x))[1:1]
            res = af.split(",")
            vector = [eval(x) for x in res]
            v = np.array(vector)
            dist = (codewords - v) ** 2
            dist = np.sum(dist, axis=1)
            dist = np.sqrt(dist)
            ind = np.where(dist == dist.min())[0][0]
            histocount[ind] += 1
            count += 1

print(outfile, str(histocount)[1:1])
print(counter)
histocount = [0] * len(codewords)
count = 0
counter += 1

#End sampling
outfile.close()
timed = time.time() - t0
print('Finished in:', timed, 'seconds')


