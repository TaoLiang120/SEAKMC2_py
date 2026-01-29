##source code mlip-2
git clone https://gitlab.com/asolovykh/mlip-2.git
##source code interface-lammps-mlip-2
git clone https://gitlab.com/ashapeev/interface-lammps-mlip-2.git

noted that the source codes have been recently updated


module purge

module load anaconda3
source $ANACONDA3_SH

conda create -p your_venv python=3.12
conda activate your_venv
conda install -c conda-forge pymatgen


### install lammps needed compilers to conda ###
##conda install -c conda-forge cmake openmpi boost pybind11 fftw compilers libblas liblapack gfortran blas lapack
##if openmpi is installed, you need to add --mpi=pmix in your submission file
conda install -c conda-forge cmake boost pybind11 fftw compilers libblas liblapack gfortran blas lapack
conda install -c conda-forge mpi4py=3.1.6

Inside mlip-2
1. File configure:
   - change ".a" to ".so"
2. File Makefile: 
   - change ".a" to ".so"
   - add " -rdynamic" to "CPPFLAGS += -fPIC"
   - add " -rdynamic" to "CXXFLAGS += -fPIC"
   - add " -rdynamic -shared" to "FFLAGS += -fPIC"
3. follow the instruction to install mlp and libinterface

Inside mlip-2 interface
1. File preinstall.sh
   - change ".a" to ".so"
   - add following lines
      make no-user-mlip
      make yes-user-mlip
      make yes-manybody
      make yes-phonon
      make yes-molecule
2. File install.sh
   - change ".a" to ".so"
   - commentted make mpi-stubs
   - change line "make $TARGET -lgfortran" to "make mode=shared $TARGET -lgfortran"


cd lammps/src/
cp liblammps.so your_venv/lib  
##export LD_LIBRARY_PATH="yourlammps/src:$LD_LIBRARY_PATH"
make install-python

conda install openkim-models


if install openmpi before lammps compilation, i.e. "conda install -c conda-forge openmpi", in run script:

srun --mpi=pmix -n ncpu your_application

otherwise, 

srun -n ncpu your_application
