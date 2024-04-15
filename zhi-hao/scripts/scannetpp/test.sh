# ns-train neus-facto --vis wandb \
#     --viewer.websocket-port 7006 \
#     --output-dir outputs/scannetpp --experiment-name 0414_office3 \
#     --trainer.steps-per-save 1000 \
#     --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
#     --trainer.max-num-iterations 240000 --trainer.steps-per-eval-batch 5000 \
#     --trainer.load-dir /hdd/sdfstudio/outputs/scannetpp/0414_office3/neus-facto/2024-04-14_180304/sdfstudio_models \
#     --pipeline.model.sdf-field.inside-outside True \
#     --pipeline.model.interlevel-loss-mult 0.0 \
#     nerfstudio-data --data /hdd/datasets/scannetpp/data/4a1a3a7dc5/dslr \
#     --center-poses False --orientation-method none --auto-scale-poses False 

# mesh extraction
ns-extract-mesh --load-config /hdd/sdfstudio/outputs/scannetpp/0414_office3/neus-facto/2024-04-14_181011/config.yml \
    --output-path /hdd/sdfstudio/outputs/scannetpp/0414_office3/neus-facto/2024-04-14_181011/mesh.ply \
    --bounding-box-min -2.0 -2.0 -2.0 --bounding-box-max 2.0 2.0 2.0 \
    --resolution 2048 --marching_cube_threshold 0.001 --simplify-mesh True