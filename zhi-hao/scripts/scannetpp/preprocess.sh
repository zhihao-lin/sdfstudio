#!/bin/bash
SCANNETPP_ROOT='/hdd/datasets/scannetpp'
OMNIDATA_ROOT='/hdd/omnidata'
SDFSTUDIO_ROOT='/hdd/sdfstudio'


SCNENES=(
    '7e09430da7' '7eac902fd5' '45b0dac5e3' '036bce3393' '49a82360aa' '9859de300f'
)


for SCENE in "${SCNENES[@]}"; do
    echo "============== $SCENE =============="
    SCANNETPP_DATA_DIR="${SCANNETPP_ROOT}/data/${SCENE}/dslr"
    # normalize poses & resize images
    python zhi-hao/scripts/scannetpp/process_scannetpp_to_psdf.py --id $SCENE

    # predict normal
    cd $OMNIDATA_ROOT
    conda activate omnidata

    python -m omnidata_tools.torch.demo_sdfstudio \
      --task normal \
      --source_dir "${SCANNETPP_ROOT}/data/${SCENE}/psdf/images" \
      --output_dir "${SCANNETPP_ROOT}/data/${SCENE}/psdf/normal"

    cd $SDFSTUDIO_ROOT
    conda activate sdfstudio

    rm ${SCANNETPP_ROOT}/data/${SCENE}/psdf/normal/*.npy
done
