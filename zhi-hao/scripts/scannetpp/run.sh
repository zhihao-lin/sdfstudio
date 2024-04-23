# # bakedsdf
# python scripts/train.py bakedsdf --vis wandb \
#     --output-dir outputs/scannetpp --experiment-name 240422_office3_bakedsdf-non-normalized-contraction-inf \
#     --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
#     --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
#     --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
#     --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
#     --machine.num-gpus 1 \
#     --pipeline.model.mono-normal-loss-mult 0.1 \
#     panoptic-data \
#     --data ../datasets/scannetpp/data/4a1a3a7dc5/psdf/ \
#     --panoptic_data False --mono_normal_data True --panoptic_segment False \
#     --orientation-method none --center-poses False --auto-scale-poses False

# # monosdf
# python scripts/train.py monosdf --vis wandb \
#     --output-dir outputs/scannetpp --experiment-name 240422_office3_monosdf-non-normalized \
#     --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
#     --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
#     --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
#     --pipeline.model.eikonal-loss-mult 0.01 \
#     --machine.num-gpus 1 \
#     --pipeline.model.mono-normal-loss-mult 0.1 \
#     panoptic-data \
#     --data ../datasets/scannetpp/data/4a1a3a7dc5/psdf/ \
#     --panoptic_data False --mono_normal_data True --panoptic_segment False \
#     --orientation-method none --center-poses False --auto-scale-poses False

# neuralangelo
python scripts/train.py neuralangelo --vis wandb \
    --output-dir outputs/scannetpp --experiment-name 240422_office3_neuralangelo-non-normalized \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 500001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data ../datasets/scannetpp/data/4a1a3a7dc5/psdf/ \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False

# python scripts/extract_mesh.py --load-config outputs/scannetpp/240417_office3_bakedsdf-hongchi-merge-delta/bakedsdf/2024-04-17_231030/config.yml \
#     --output-path outputs/scannetpp/240417_office3_bakedsdf-hongchi-merge-delta/bakedsdf/2024-04-17_231030/mesh.ply \
#     --bounding-box-min -2.0 -2.0 -2.0 --bounding-box-max 2.0 2.0 2.0 \
#     --resolution 2048 --marching_cube_threshold 0.001 --create_visibility_mask True --simplify-mesh True