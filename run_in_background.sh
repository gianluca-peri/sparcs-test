#!/bin/bash
# Run script train_models.py in background (the -u is necessary to avoid output buffering)
rm train.log 2>/dev/null
nohup python -u $1 > train.log 2>&1 &