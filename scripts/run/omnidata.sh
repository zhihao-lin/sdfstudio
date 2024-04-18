python scripts/omnidata/normal_prior.py \
  --source_dir /home/hongchix/scratch/scannetpp/2e67a32314/images \
  --output_dir /home/hongchix/scratch/scannetpp/2e67a32314/normal/ \
  --ckpt_dir /home/hongchix/main/codes/omnidata/omnidata_tools/torch/pretrained_models/ \
  --slice 1

python scripts/omnidata/depth_prior.py \
    --source_dir /home/hongchix/main/root/datasets/scannet/scannetpp/data/c50d2d1d42/dslr/plift/images \
    --output_dir /home/hongchix/main/root/datasets/scannet/scannetpp/data/c50d2d1d42/dslr/plift/depth \
    --vis_dir /home/hongchix/main/root/datasets/scannet/scannetpp/data/c50d2d1d42/dslr/plift/depth_vis \
    --ckpt_dir /home/hongchix/main/codes/omnidata/omnidata_tools/torch/pretrained_models/