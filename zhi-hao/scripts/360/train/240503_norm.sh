# counter-norm
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240503_counter_bakedsdf-mlp-norm \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/colmap_sdfstudio/counter \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# bicycle-norm
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240503_bicycle_bakedsdf-mlp-norm \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm l2\
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/colmap_sdfstudio/bicycle \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# bonsai-norm
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240503_bonsai_bakedsdf-mlp-norm \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    panoptic-data \
    --data ../datasets/colmap_sdfstudio/bonsai \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# kitchen-norm
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240503_kitchen_bakedsdf-mlp-norm \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    panoptic-data \
    --data ../datasets/colmap_sdfstudio/kitchen \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# room-norm
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240503_room_bakedsdf-mlp-norm \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 \
    panoptic-data \
    --data ../datasets/colmap_sdfstudio/room \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# stump-norm
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/colmap_studio --experiment-name 240503_stump_bakedsdf-mlp-norm \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm l2 \
    panoptic-data \
    --data ../datasets/colmap_sdfstudio/stump \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False