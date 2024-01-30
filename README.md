# Shallow-Moments-Simulation



## Install

### Install python libraries
```
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
echo "export PYTHONPATH='${PYTHONPATH}:$(pwd)'" >> .venv/bin/activate
echo "export SMPYTHON='$(pwd)'" >> .venv/bin/activate
```

### Install C libraries
```
cd install
./configure
echo "export VOLKOSPATH=/home/ingo/git/SMM/SMS" >> .venv/bin/activate
```

```
cd ../library/solver
make
```

### Dependencies (Debian)
```
sudo apt install wget
sudo apt install make
sudo apt install g++
sudo apt install python3-dev 
sudo apt install mpich
<!-- for kokkos -->
sudo apt install cmake
sudo apt install bc
```

## Changes on the sympy files:
- .c to .cpp
- using namespace sympy (or the source and source_jacobian function is named twice)
- inline? const? double* to kokkos view?

## Open questions:
- Are the arrays initiaized with zeros -> if not, do so. If yes, remove one part where I do it.
- Boundary conditions for Qaux
- lambdas in space operator as classes?
- nc-flux?
- dynamic includes (see todo)
- adaptive time stepping
- source term
 

