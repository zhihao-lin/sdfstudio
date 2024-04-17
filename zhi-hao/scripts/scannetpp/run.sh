# ns-train bakedsdf --vis wandb \
#     --output-dir outputs/scannetpp --experiment-name 0415_office3_sdf_mono_bakedsdf_no_pose_norm \
#     --trainer.steps-per-save 1000 \
#     --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
#     --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
#     --pipeline.model.sdf-field.inside-outside True \
#     --pipeline.model.mono-depth-loss-mult 0.1 \
#     --pipeline.model.mono-normal-loss-mult 0.05 \
#     --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.sdf-field.bias 1.5 \
#     --machine.num-gpus 1 --pipeline.model.scene-contraction-norm l2 \
#     sdfstudio-data --data ../datasets/scannetpp/data/4a1a3a7dc5/sdfstudio \
#     --include-mono-prior True \
#     --center-poses False --orientation-method none --auto-scale-poses False 

# mesh extraction
ns-extract-mesh --load-config /hdd/sdfstudio/outputs/scannetpp/0415_office3_sdf_mono/bakedsdf/2024-04-15_125038/config.yml \
    --output-path /hdd/sdfstudio/outputs/scannetpp/0415_office3_sdf_mono/bakedsdf/2024-04-15_125038/mesh.ply \
    --bounding-box-min -1.0 -1.0 -1.0 --bounding-box-max 1.0 1.0 1.0 \
    --resolution 2048 --marching_cube_threshold 0.001 --simplify-mesh True