from mpi4py import MPI
from lammps import lammps
lmp = lammps()
lmp.file("in.lammps")
