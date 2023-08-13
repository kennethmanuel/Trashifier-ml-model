#!/bin/bash

# Change directory to the parent folder containing the subfolders
cd experiment

# Loop through each subfolder
for folder in eid_*/; do
    # Enter the subfolder
    cd "$folder"
    
    # Execute the train.py script
    python train.py
    
    # Return to the parent folder
    cd ..
done