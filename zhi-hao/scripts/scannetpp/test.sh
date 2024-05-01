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

# storage
python scripts/train.py bakedsdf-mlp --vis wandb \
    --output-dir outputs/scannetpp --experiment-name 240429_storage_bakedsdf-mlp \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/scannetpp/data/0a5c013435/psdf/ \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# office
python scripts/train.py bakedsdf-mlp --vis wandb \
    --output-dir outputs/scannetpp --experiment-name 240429_office_bakedsdf-mlp \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/scannetpp/data/0b031f3119/psdf/ \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# room
python scripts/train.py bakedsdf-mlp --vis wandb \
    --output-dir outputs/scannetpp --experiment-name 240429_room_bakedsdf-mlp \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/scannetpp/data/0a7cc12c0e/psdf/ \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# office2
python scripts/train.py bakedsdf-mlp --vis wandb \
    --output-dir outputs/scannetpp --experiment-name 240429_office2_bakedsdf-mlp \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/scannetpp/data/1ada7a0617/psdf/ \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# empty
python scripts/train.py bakedsdf-mlp --vis wandb \
    --output-dir outputs/scannetpp --experiment-name 240429_empty_bakedsdf-mlp \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/scannetpp/data/1c4b893630/psdf/ \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# lab
python scripts/train.py bakedsdf-mlp --vis wandb \
    --output-dir outputs/scannetpp --experiment-name 240429_lab_bakedsdf-mlp \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/scannetpp/data/1d003b07bd/psdf/ \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# attic
python scripts/train.py bakedsdf-mlp --vis wandb \
    --output-dir outputs/scannetpp --experiment-name 240429_attic_bakedsdf-mlp \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/scannetpp/data/3db0a1c8f3/psdf/ \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# attic2
python scripts/train.py bakedsdf-mlp --vis wandb \
    --output-dir outputs/scannetpp --experiment-name 240429_attic2_bakedsdf-mlp \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/scannetpp/data/3f1e1610de/psdf/ \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# engine
python scripts/train.py bakedsdf-mlp --vis wandb \
    --output-dir outputs/scannetpp --experiment-name 240429_engine_bakedsdf-mlp \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/scannetpp/data/5fb5d2dbf2/psdf/ \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# lab2
python scripts/train.py bakedsdf-mlp --vis wandb \
    --output-dir outputs/scannetpp --experiment-name 240429_lab2_bakedsdf-mlp \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/scannetpp/data/7cd2ac43b4/psdf/ \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False