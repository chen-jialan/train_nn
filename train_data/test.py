# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 00:49:34 2022

@author: jlchen
"""

import os
from NN_train import NSDS_algorithm
import ase.io.vasp
from ase.io.trajectory import Trajectory
from ase.db import connect
import numpy as np

filename = 'all_candidates.traj'
filetype = 'traj'
atomtype1 = ['Pt','Fe','O']
atomtype2 = ['Pt','Fe','O']
nn='EANN_PES_DOUBLE4.pt'
energy_criterion = 2

da = Trajectory(filename) 
energy = np.array([da[i].get_potential_energy() for i in range(len(da))])
np.save('energy1.npy',energy)

atom_nn_traj = NSDS_algorithm(filename=filename,atomtype1=atomtype1,atomtype2=atomtype2,nn2=nn,
                              energy_criterion=energy_criterion)
atoms_all = atom_nn_traj.extreme_point()
if os.path.exists('./nn_train.db'):
    os.remove('./nn_train.db')
for atoms in atoms_all:
    print(atoms.get_potential_energy())
    db = connect('nn_train.db')
    db.write(atoms)

#ase.io.write('POSCAR-final', atom_nn_traj.extreme_point(), format='vasp', vasp5='True')
"""
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.stats as stats
a = np.load('energy1.npy')
#b = np.load('energy2.npy')
b =  abs(np.load('F.npy'))
plt.plot(a-min(a), b, 'g*-',label='original values')
#print(((b-a)**2)[signal.argrelextrema((b-a)**2, np.greater)])
#print(signal.argrelextrema((b-a)**2, np.greater))
z1 = np.polyfit(a-min(a),b, 100)
p1 = np.poly1d(z1) #多项式系数
yvals=p1(a-min(a)) 
peaks = signal.find_peaks(yvals, distance=5)[0]
prominences = signal.peak_prominences(yvals, peaks)[0]
print(prominences)
contour_heights = yvals[peaks] - prominences
plt.plot(a-min(a), yvals, 'r',label='polyfit values')
plt.vlines(x=(a-min(a))[peaks], ymin=contour_heights, ymax=yvals[peaks])
#plt.plot((a-min(a))[signal.argrelextrema((b-a)**2,np.greater,order=1)[0]],((b-a)**2)[signal.argrelextrema(abs(b-a)**2, np.greater,order=1)],'o')
plt.show()
"""
