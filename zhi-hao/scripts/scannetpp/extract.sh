python scripts/extract_mesh.py --load-config outputs/scannetpp/240422_office3_bakedsdf-non-normalized-contraction-inf/bakedsdf/2024-04-23_164817/config.yml \
    --output-path outputs/scannetpp/240422_office3_bakedsdf-non-normalized-contraction-inf/bakedsdf/2024-04-23_164817/mesh.ply \
    --bounding-box-min -2.0 -2.0 -2.0 --bounding-box-max 2.0 2.0 2.0 \
    --resolution 2048 --marching_cube_threshold 0.001 --create_visibility_mask True --simplify-mesh True