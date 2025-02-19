docker run \
    --mount type=bind,source='/home/ml4u/BKTeam/Tr/ImageNet',target=/root/data/ImageNet \
    --mount type=bind,source='/dev/shm/Places2',target=/root/data/Places2\
    --mount type=bind,source='/media/ml4u/Extreme SSD/datasets/celebA-HQ',target=/root/data/CelebA-HQ\
    --mount type=bind,source='/media/ml4u/Extreme SSD/datasets/FFHQ',target=/root/data/FFHQ \
    --mount type=bind,source='/media/ml4u/Extreme SSD/log/TRUNG/DifFace_Thesis',target=/root/source \
    --mount type=bind,source='/media/ml4u/Extreme SSD/log/TRUNG/mask',target=/root/data/Mask \
    --mount type=bind,source='/media/ml4u/Extreme SSD/log/TRUNG/log',target=/root/log \
    --gpus all -d -it --privileged --shm-size=48GB    hoaitrung/blind_inpainting


# export CUDA_VISIBLE_DEVICES="1"
# cd source
# export WANDB_API_KEY = ""
# torchrun --standalone --nproc_per_node=1 --nnodes=1 main.py --cfg_path configs/training/predicted_edge_SwinUnet_docker.yaml --save_dir /root/log 