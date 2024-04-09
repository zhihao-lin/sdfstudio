# training
ns-train bakedsdf --vis wandb \
    --data data/colmap_sdfstudio/garden \
    --output-dir outputs/garden --experiment-name 240404_garden_bakedsdf-colmap_sdfstudio \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm l2 \
    mipnerf360-data --data /hdd/sdfstudio/data/colmap_sdfstudio/garden \
    --center-poses False --orientation-method none --auto-scale-poses False
# --trainer.load-dir /hdd/sdfstudio/outputs/garden/240327_bakedsdf-mlp_colmap_ns/bakedsdf-mlp/2024-03-27_020413/sdfstudio_models \

# mesh extraction
ns-extract-mesh --load-config /hdd/sdfstudio/outputs/garden/240404_garden_bakedsdf-colmap_sdfstudio/bakedsdf/2024-04-05_010522/config.yml \
    --output-path /hdd/sdfstudio/outputs/garden/240404_garden_bakedsdf-colmap_sdfstudio/bakedsdf/2024-04-05_010522/mesh.ply \
    --bounding-box-min -2.0 -2.0 -2.0 --bounding-box-max 2.0 2.0 2.0 \
    --resolution 2048 --marching_cube_threshold 0.001 --create_visibility_mask True --simplify-mesh True

# rendering
# ns-render-mesh --meshfile meshes/bakedsdf-mlp-garden-4096.ply --traj ellipse --fps 60 --num_views 480 --output_path renders/garden.mp4 mipnerf360-data --data data/nerfstudio-data-mipnerf360/garden

# Bake texture
python scripts/texture.py \
    --load-config /hdd/sdfstudio/outputs/garden/240404_garden_bakedsdf-colmap_sdfstudio/bakedsdf/2024-04-05_010522/config.yml \
    --input-mesh-filename /hdd/sdfstudio/outputs/garden/240404_garden_bakedsdf-colmap_sdfstudio/bakedsdf/2024-04-05_010522/mesh-simplify.ply \
    --output-dir /hdd/sdfstudio/outputs/garden/240404_garden_bakedsdf-colmap_sdfstudio/bakedsdf/2024-04-05_010522/ \
    --target_num_faces None