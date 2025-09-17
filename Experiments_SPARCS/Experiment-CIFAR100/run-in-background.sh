#!/bin/bash

# Run the Python script in the background
nohup python3 Experiment-2/SPARCS-with-backbone.py > Experiment-2/log.txt 2>&1 &

# Print the process ID of the background job
echo "SPARCS-with-backbone.py is running in the background with PID $!"