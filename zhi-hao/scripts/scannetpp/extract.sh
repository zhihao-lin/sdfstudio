scenes=(
    '/hdd/sdfstudio/outputs/scannetpp/240429_bathroom_bakedsdf-mlp/bakedsdf-mlp/2024-05-01_115823'
    '/hdd/sdfstudio/outputs/scannetpp/240429_bathroom2_bakedsdf-mlp/bakedsdf-mlp/2024-05-01_115823'
    '/hdd/sdfstudio/outputs/scannetpp/240429_game_bakedsdf-mlp/bakedsdf-mlp/2024-05-01_115848'
    '/hdd/sdfstudio/outputs/scannetpp/240429_kitchen_bakedsdf-mlp/bakedsdf-mlp/2024-05-01_121846'
    '/hdd/sdfstudio/outputs/scannetpp/240429_room2_bakedsdf-mlp/bakedsdf-mlp/2024-05-01_121814'
    '/hdd/sdfstudio/outputs/scannetpp/240429_room3_bakedsdf-mlp/bakedsdf-mlp/2024-05-01_120411'
    '/hdd/sdfstudio/outputs/scannetpp/240429_empty_bakedsdf-mlp/bakedsdf-mlp/2024-05-01_154020'
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