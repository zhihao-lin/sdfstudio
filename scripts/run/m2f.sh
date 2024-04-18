DATA="/home/hongchix/main/root/datasets/scannet/scannetpp/data/c50d2d1d42/dslr/"
mkdir -p ${DATA}/panoptic
python demo.py --config-file ../configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
    --input ${DATA}/perspective \
    --output ${DATA}/panoptic \
    --predictions ${DATA}/panoptic \
    --opts MODEL.WEIGHTS ../checkpoints/model_final_f07440.pkl
