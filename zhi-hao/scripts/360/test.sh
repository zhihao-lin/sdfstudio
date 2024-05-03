DATA_ROOT='/hdd/sdfstudio/data/colmap_sdfstudio/garden_4'

python zhi-hao/scripts/360/orient.py \
    --transform $DATA_ROOT/transforms.json \
    --normal $DATA_ROOT/normal \
    --mask $DATA_ROOT/mask 
