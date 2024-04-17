python scripts/train.py bakedsdf --vis wandb \
    --output-dir outputs/scannetpp --experiment-name 240417_office3_bakedsdf-hongchi \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm l2 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data /hdd/datasets/scannetpp/data/4a1a3a7dc5/psdf \
    --panoptic_data False \
    --mono_normal_data True \
    --panoptic_segment False


# python scripts/extract_mesh.py --load-config ./outputs/scannetpp/240415_scannetpp_bakedsdf-colmap_sdfstudio_normal_mono/bakedsdf/2024-04-15_094639/config.yml \
#     --output-path ./outputs/scannetpp/240415_scannetpp_bakedsdf-colmap_sdfstudio_normal_mono/bakedsdf/2024-04-15_094639/mesh.ply \
#     --bounding-box-min -2.0 -2.0 -2.0 --bounding-box-max 2.0 2.0 2.0 \
#     --resolution 2048 --marching_cube_threshold 0.001 --create_visibility_mask True --simplify-mesh True