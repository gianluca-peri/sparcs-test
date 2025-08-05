#!/bin/bash

# Run the Python script in the background
nohup python3 Experiment-1/SPARCS-in-CNN.py > Experiment-1/log.txt 2>&1 &

# Print the process ID of the background job
echo "SPARCS-in-CNN.py is running in the background with PID $!"