#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: docker run --rm -v \$(pwd):/workspace basilisk_runner <case_file.c>"
    exit 1
fi

FILE=$1
BASENAME="${FILE%.*}"
LOGFILE="${BASENAME}.log"

# Clean previous run artifacts
> "$LOGFILE"
rm -f *_series.log
rm -f *.vtk.series

echo "==== [1/3] Compiling $FILE ====" | tee -a "$LOGFILE"
qcc -Wall -O2 -disable-dimensions "$FILE" -o "$BASENAME" -lm 2>&1 | tee -a "$LOGFILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "==== [2/3] Compilation successful. Running $BASENAME ====" | tee -a "$LOGFILE"
    ./"$BASENAME" 2>&1 | tee -a "$LOGFILE"
    echo "==== [3/3] Simulation complete. Output saved to workspace. ====" | tee -a "$LOGFILE"
else
    echo "==== [!] Compilation failed. Check $LOGFILE for details. ====" | tee -a "$LOGFILE"
    exit 1
fi
