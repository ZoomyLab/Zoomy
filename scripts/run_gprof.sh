#!/bin/sh

cd bin
# gprof volkos | python -m gprof2dot | dot -Tpng -o ../profiling_result.png
gprof volkos | python -m gprof2dot -n 1 -e 1 -w -s | dot -Tpng -o ../profiling_result.png
cd ..
