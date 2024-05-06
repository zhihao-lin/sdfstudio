# garden
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240501_garden_bakedsdf-mlp \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm l2 \
    mipnerf360-data --data ../datasets/colmap_sdfstudio/garden_4 \
    --center-poses False --orientation-method none --auto-scale-poses False

# garden-pan
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240501_garden_bakedsdf-mlp-pan \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm l2 \
    panoptic-data \
    --data ../datasets/colmap_sdfstudio/garden_4 \
    --panoptic_data False --mono_normal_data False --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# counter
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240501_counter_bakedsdf-mlp \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    mipnerf360-data --data ../datasets/colmap_sdfstudio/counter \
    --center-poses False --orientation-method none --auto-scale-poses False


# bicycle
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240503_bicycle_bakedsdf-mlp \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm l2\
    mipnerf360-data --data ../datasets/colmap_sdfstudio/bicycle \
    --center-poses False --orientation-method none --auto-scale-poses False

# bonsai
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240503_bonsai_bakedsdf-mlp \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    mipnerf360-data --data ../datasets/colmap_sdfstudio/bonsai \
    --center-poses False --orientation-method none --auto-scale-poses False

# kitchen
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240503_kitchen_bakedsdf-mlp \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    mipnerf360-data --data ../datasets/colmap_sdfstudio/kitchen \
    --center-poses False --orientation-method none --auto-scale-poses False

# room
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240503_room_bakedsdf-mlp \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    mipnerf360-data --data ../datasets/colmap_sdfstudio/room \
    --center-poses False --orientation-method none --auto-scale-poses False

# stump
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240503_stump_bakedsdf-mlp \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm l2 \
    mipnerf360-data --data ../datasets/colmap_sdfstudio/stump \
    --center-poses False --orientation-method none --auto-scale-poses False