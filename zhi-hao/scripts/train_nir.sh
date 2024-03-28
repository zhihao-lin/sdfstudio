ns-train monosdf --experiment-name office_1 \
    --pipeline.model.sdf-field.geometric-init True \
    --pipeline.model.sdf-field.bias 0.8 \
    --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.mono-depth-loss-mult 0.05 \
    --pipeline.model.mono-normal-loss-mult 0.05 \
    --vis viewer \
    sdfstudio-data --data /hdd/datasets/nir/office_1/sdfstudio/ \
    --include_mono_prior True
    

ns-extract-mesh \
    --load-config outputs/office_1/monosdf/2023-09-04_181500/config.yml \
    --output-path outputs/office_1/monosdf/2023-09-04_181500/mesh.ply