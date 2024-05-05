scenes=(
    '/hdd/sdfstudio/outputs/colmap_studio/240503_bicycle_bakedsdf-mlp/bakedsdf-mlp/2024-05-03_113748'
    '/hdd/sdfstudio/outputs/colmap_studio/240503_bonsai_bakedsdf-mlp/bakedsdf-mlp/2024-05-03_113823'
    '/hdd/sdfstudio/outputs/colmap_studio/240503_kitchen_bakedsdf-mlp/bakedsdf-mlp/2024-05-03_215540'
    '/hdd/sdfstudio/outputs/colmap_studio/240503_room_bakedsdf-mlp/bakedsdf-mlp/2024-05-03_113856'
    '/hdd/sdfstudio/outputs/colmap_studio/240503_stump_bakedsdf-mlp/bakedsdf-mlp/2024-05-03_113856'
)

for scene in "${scenes[@]}"; do
    echo "============== $scene =============="
    # extract mesh
    python scripts/extract_mesh.py --load-config $scene/config.yml \
        --output-path $scene/mesh.ply \
        --bounding-box-min -2.0 -2.0 -2.0 --bounding-box-max 2.0 2.0 2.0 \
        --resolution 2048 --marching_cube_threshold 0.001 --create_visibility_mask True --simplify-mesh True

    mkdir $scene/textured
    # Bake texture
    python scripts/texture.py \
        --load-config $scene/config.yml \
        --input-mesh-filename $scene/mesh-simplify.ply \
        --output-dir $scene/textured \
        --target_num_faces None
done