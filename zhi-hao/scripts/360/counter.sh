# training
ns-train bakedsdf --vis wandb \
    --output-dir outputs/counter --experiment-name 240405_counter_bakedsdf-colmap_sdfstudio \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm l2 \
    mipnerf360-data --data data/colmap_sdfstudio/counter \
    --center-poses False --orientation-method none --auto-scale-poses False
# --trainer.load-dir /projects/perception/personals/zhihao/sdfstudio/outputs/counter/240327_bakedsdf-mlp_colmap_sdf/bakedsdf-mlp/2024-03-27_024121/sdfstudio_models\


# mesh extraction
ns-extract-mesh --load-config /hdd/sdfstudio/outputs/counter/240405_counter_bakedsdf-colmap_sdfstudio/bakedsdf/2024-04-05_111134/config.yml \
    --output-path /hdd/sdfstudio/outputs/counter/240405_counter_bakedsdf-colmap_sdfstudio/bakedsdf/2024-04-05_111134/mesh.ply \
    --bounding-box-min -2.0 -2.0 -2.0 --bounding-box-max 2.0 2.0 2.0 \
    --resolution 2048 --marching_cube_threshold 0.001 --create_visibility_mask True --simplify-mesh True