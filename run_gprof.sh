#!/bin/sh

cd bin
gprof volkos | python -m gprof2dot | dot -Tpng -o ../profiling_result.png
cd ..