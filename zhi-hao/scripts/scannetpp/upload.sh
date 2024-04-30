#!/bin/bash

scenes=('0a5c013435' '0b031f3119' '0a7cc12c0e' '1ada7a0617' '1c4b893630' '1d003b07bd' '3db0a1c8f3' '3f1e1610de' '5fb5d2dbf2' '7cd2ac43b4')

for scene in "${scenes[@]}"; do
    echo "============== Uploading $scene =============="
    scp -r /hdd/datasets/scannetpp/data/$scene/psdf campus-cluster:/projects/perception/datasets/scannetpp/data/$scene
done