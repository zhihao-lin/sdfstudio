scenes=(
    '/hdd/sdfstudio/outputs/scannetpp/240429_lab2_bakedsdf-mlp/bakedsdf-mlp/2024-04-29_122848'
    '/hdd/sdfstudio/outputs/scannetpp/240429_office_bakedsdf-mlp/bakedsdf-mlp/2024-04-29_194612'
    '/hdd/sdfstudio/outputs/scannetpp/240429_storage_bakedsdf-mlp/bakedsdf-mlp/2024-04-29_124328'
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