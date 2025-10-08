#!/bin/bash

# export HOME=$HOME
# export PATH=/usr/bin:/bin

# Load OpenFOAM environment
source /home/ingo/Git/OpenFOAM/OpenFOAM-12/etc/bashrc
ls
echo $WM_PROJECT_USER_DIR
echo $WM_PROJECT_VERSION
echo $USER

wclean
wmake

