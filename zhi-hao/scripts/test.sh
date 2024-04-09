# Download some test data: you might need to install curl if your system don't have that
ns-download-data sdfstudio

# Train model on the dtu dataset scan65
ns-train neus-facto \
    --pipeline.model.sdf-field.inside-outside False \
    --vis viewer --experiment-name neus-facto-dtu65 \
    sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65

# Or you could also train model on the Replica dataset room0 with monocular priors
ns-train neus-facto \
    --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.mono-depth-loss-mult 0.1 \
    --pipeline.model.mono-normal-loss-mult 0.05 \
    --vis viewer --experiment-name neus-facto-replica1 \
    sdfstudio-data --data data/sdfstudio-demo-data/replica-room0 --include_mono_prior True