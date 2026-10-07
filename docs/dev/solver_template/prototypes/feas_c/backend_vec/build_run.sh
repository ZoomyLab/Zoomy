#!/bin/sh
# Build + run the emitted march for each variant against the vector backend.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
for v in ${@:-o1_plain o2_plain o1_mood o2_mood o2_mood_coupled o2_all o1_dtfloor}; do
  D="$HERE/../fixed/$v"
  g++ -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -I "$HERE/../fixed" -I "$D" \
      "$HERE/main.cpp" "$HERE/bodies_swe.cpp" -o "$D/march_$v"
  echo "== $v"; "$D/march_$v" "$D/final_state.txt"
done
