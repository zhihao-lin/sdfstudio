# counter-norm
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240505_counter_norm_0 \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/colmap_sdfstudio/counter \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# bicycle-norm
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240505_bicycle_norm_0 \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/colmap_sdfstudio/bicycle \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# bonsai-norm
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240505_bonsai_norm_0 \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/colmap_sdfstudio/bonsai \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# kitchen-norm
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240505_kitchen_norm_0 \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/colmap_sdfstudio/kitchen \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# room-norm
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240505_room_norm_0 \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/colmap_sdfstudio/room \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# stump-norm
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240505_stump_norm_0 \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/colmap_sdfstudio/stump \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False