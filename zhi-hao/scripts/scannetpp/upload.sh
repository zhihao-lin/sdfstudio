#!/bin/bash

scenes=(
    '7e09430da7' 
    '7eac902fd5' 
    '45b0dac5e3' 
    '036bce3393' 
    '49a82360aa' 
    '9859de300f')

for scene in "${scenes[@]}"; do
    echo "============== Uploading $scene =============="
    # scp -r /hdd/datasets/scannetpp/data/$scene/psdf campus-cluster:/projects/perception/datasets/scannetpp/data/$scene
    scp -r /hdd/datasets/scannetpp/data/$scene/psdf delta:/scratch/bcrp/cl121/datasets/scannetpp/data/$scene
done