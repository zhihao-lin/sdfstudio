#!/bin/bash
SCANNETPP_ROOT='/hdd/datasets/scannetpp'
SCENE='7cd2ac43b4'
SCANNETPP_DATA_DIR="${SCANNETPP_ROOT}/data/${SCENE}/dslr"
OMNIDATA_ROOT='/hdd/omnidata'
SDFSTUDIO_ROOT='/hdd/sdfstudio'

python zhi-hao/scripts/scannetpp/process_scannetpp_to_psdf.py --id $SCENE

# Extract monocular prior with Omnidata
cd $OMNIDATA_ROOT
conda activate omnidata

python -m omnidata_tools.torch.demo_sdfstudio \
  --task normal \
  --source_dir "${SCANNETPP_ROOT}/data/${SCENE}/psdf/images" \
  --output_dir "${SCANNETPP_ROOT}/data/${SCENE}/psdf/normal"

cd $SDFSTUDIO_ROOT
conda activate sdfstudio

# get meta_data.json 
# python zhi-hao/scripts/scannetpp/process_nerfstudio_to_sdfstudio.py \
#     --data  $SCANNETPP_DATA_DIR --img_size 1024 \
#     --output-dir "${SCANNETPP_ROOT}/data/${SCENE}/sdfstudio-1024" \
#     --data-type colmap --scene-type indoor --mono-prior \
#     --omnidata-path /hdd/omnidata/omnidata_tools/torch/ \
#     --pretrained-models /hdd/omnidata/omnidata_tools/torch/pretrained_models
