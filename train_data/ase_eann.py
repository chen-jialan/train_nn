# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:52:34 2021

@author: DELL
"""

import ase.io.vasp
from ase import Atoms
from ase.calculators.eann import EANN
from ase.calculators.reann import REANN
#import os
#import re
from ase.optimize.minimahopping import MinimaHopping
from ase.optimize.minimahopping import MHPlot
from ase.optimize import LBFGS
from ase.io.trajectory import Trajectory
import sys
from ase.vibrations import Vibrations
import numpy as np

#-------------------------LBDGS----------------------------
poscar = sys.argv[1] 
cell1 = ase.io.vasp.read_vasp("%s" %poscar)
atoms = Atoms(cell1)
atoms.set_positions = atoms.positions
#print(atoms.positions)
atomtype = ['Pt','Fe','O']
#atomtype = ['Cu','Ce','O','C']
device='cpu'
period=[1,1,1]
nn = 'EANN_PES_DOUBLE.pt'
#atoms.calc = EANN(device=device,atomtype=atomtype,period=period,nn = nn)
atoms.calc = REANN(atomtype=atomtype,period=[1,1,0],nn = 'REANN_PES_DOUBLE.pt')
#print(atoms.positions)
dyn = LBFGS(atoms,trajectory='atom2.traj')
dyn.run(fmax=0.1)
#traj = Trajectory('atom2.traj')
#atoms = traj[-1]
#-----------------------write final data-----------------
traj = Trajectory('atom2.traj')
atoms = traj[-1]
ase.io.write('POSCAR-final', atoms, format='vasp', vasp5='True')
#print(atoms.get_potential_energy())
#print(atoms)
#e= atoms.get_potential_energy()
#f = atoms.get_forces()
#print(e,f)
