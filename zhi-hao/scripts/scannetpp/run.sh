ns-train neus-facto --vis wandb \
    --viewer.websocket-port 7006 \
    --output-dir outputs/scannetpp --experiment-name 0414_office3_sdf_mono \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.mono-depth-loss-mult 0.1 \
    --pipeline.model.mono-normal-loss-mult 0.05 \
    sdfstudio-data --data /hdd/datasets/scannetpp/data/4a1a3a7dc5/sdfstudio \
    --include-mono-prior True \
    --center-poses False --orientation-method none --auto-scale-poses False 