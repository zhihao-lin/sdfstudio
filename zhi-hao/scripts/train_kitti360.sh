ns-train neuralangelo --experiment-name neuralangelo-kitti360 \
    --pipeline.model.sdf-field.inside-outside False \
    --pipeline.model.mono-depth-loss-mult 0.0 \
    --pipeline.model.mono-normal-loss-mult 0.05 \
    --vis viewer --pipeline.model.far-plane-bg 10 \
    sdfstudio-data --data data/kitti360-1538-1601 \
    --include_mono_prior True 

ns-extract-mesh \
    --load-config outputs/office_1/monosdf/2023-09-04_181500/config.yml \
    --output-path outputs/office_1/monosdf/2023-09-04_181500/mesh.ply