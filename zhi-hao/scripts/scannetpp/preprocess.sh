#!/bin/bash
SCANNETPP_ROOT='/hdd/datasets/scannetpp'
OMNIDATA_ROOT='/hdd/omnidata'
SDFSTUDIO_ROOT='/hdd/sdfstudio'


SCNENES=(
    '45b0dac5e3'
    '036bce3393'
    '0b031f3119'
    '7e09430da7'
    '49a82360aa'
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

# for SCENE in "${SCNENES[@]}"; do
#     echo "============== $SCENE =============="
#     mkdir ${SCANNETPP_ROOT}/data/${SCENE}/psdf
#     mv ${SCANNETPP_ROOT}/data/${SCENE}/images ${SCANNETPP_ROOT}/data/${SCENE}/psdf
#     mv ${SCANNETPP_ROOT}/data/${SCENE}/normal ${SCANNETPP_ROOT}/data/${SCENE}/psdf
#     mv ${SCANNETPP_ROOT}/data/${SCENE}/transforms.json ${SCANNETPP_ROOT}/data/${SCENE}/psdf
#     mv ${SCANNETPP_ROOT}/data/${SCENE}/transforms_all.json ${SCANNETPP_ROOT}/data/${SCENE}/psdf
# done
