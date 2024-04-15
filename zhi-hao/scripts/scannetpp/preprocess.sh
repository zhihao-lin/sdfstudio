#!/bin/bash
SCANNETPP_ROOT='/hdd/datasets/scannetpp'
SCENE='4a1a3a7dc5'
SCANNETPP_DATA_DIR="${SCANNETPP_ROOT}/data/${SCENE}/dslr"
OMNIDATA_ROOT='/hdd/omnidata'
SDFSTUDIO_ROOT='/hdd/sdfstudio'

# Extract monocular prior with Omnidata
# cd $OMNIDATA_ROOT
# conda activate omnidata

# python -m omnidata_tools.torch.demo_sdfstudio \
#   --task depth --mode patch --img_size 1024\
#   --source_dir $SCANNETPP_DATA_DIR/undistorted_images \
#   --output_dir $SCANNETPP_DATA_DIR/depth_omnidata

# python -m omnidata_tools.torch.demo_sdfstudio \
#   --task normal --mode patch --img_size 1024\
#   --source_dir $SCANNETPP_DATA_DIR/undistorted_images \
#   --output_dir $SCANNETPP_DATA_DIR/normal_omnidata

# cd $SDFSTUDIO_ROOT
# conda activate sdfstudio

# get meta_data.json 
python zhi-hao/scripts/scannetpp/process_nerfstudio_to_sdfstudio.py \
    --data  $SCANNETPP_DATA_DIR \
    --output-dir "${SCANNETPP_ROOT}/data/${SCENE}/sdfstudio" \
    --data-type colmap --scene-type indoor --mono-prior \
    --omnidata-path /hdd/omnidata/omnidata_tools/torch/ \
    --pretrained-models /hdd/omnidata/omnidata_tools/torch/pretrained_models
