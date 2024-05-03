OMNIDATA_ROOT='/hdd/omnidata'
SDFSTUDIO_ROOT='/hdd/sdfstudio'
DATA_ROOT='/hdd/sdfstudio/data/colmap_sdfstudio'

# SCNENES=(
#     'bicycle' 'bonsai' 'counter' 'garden_4' 'kitchen' 'room' 'stump'
# )

SCNENES=(
    'garden_4'
)


for SCENE in "${SCNENES[@]}"; do
    echo "============== $SCENE =============="

    # predict normal
    cd $OMNIDATA_ROOT
    conda activate omnidata

    python -m omnidata_tools.torch.demo_sdfstudio \
      --task normal --mode patch --img_size 768 \
      --source_dir "${DATA_ROOT}/${SCENE}/images" \
      --output_dir "${DATA_ROOT}/${SCENE}/normal"

    cd $SDFSTUDIO_ROOT
    conda activate sdfstudio

    rm ${DATA_ROOT}/${SCENE}/normal/*.npy
done