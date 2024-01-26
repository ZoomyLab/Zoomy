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


