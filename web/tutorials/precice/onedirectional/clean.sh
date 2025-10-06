#!/bin/sh
set -e -u

mv 0 ic
rm -rf 0*
rm -rf 1*
rm -rf 2*
rm -rf 3*
rm -rf 4*
rm -rf 5*
rm -rf 6*
rm -rf 7*
rm -rf 8*
rm -rf 9*
mv ic 0

rm *.log
rm *.out

cd export1
rm -rf *
cd ..
