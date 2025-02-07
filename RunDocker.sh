docker run \
    --mount type=bind,source=/home/ml4u/BKTeam/dataset/ImageNet256,target=/root/data/ImageNet \
    --mount type=bind,source=/home/ml4u/BKTeam/dataset/CelebA-HQ256,target=/root/data/CelebA-HQ\
    --mount type=bind,source=/home/ml4u/BKTeam/dataset/FFHQ256,target=/root/data/FFHQ \
    --mount type=bind,source=/media/ml4u/Samsung_T5/source/Trung/DifFace_Thesis,target=/root/source \
    --mount type=bind,source=/media/ml4u/Samsung_T5/source/Trung/test_mask,target=/root/data/Mask \
    --mount type=bind,source=/media/ml4u/Samsung_T5/source/Trung/log,target=/root/log \
    --gpus all -it --privileged --shm-size=24GB  hoaitrung/blind_inpainting


# export CUDA_VISIBLE_DEVICES="1"
# cd source
# export WANDB_API_KEY = ""
# torchrun --standalone --nproc_per_node=1 --nnodes=1 main.py --cfg_path configs/training/predicted_edge_SwinUnet_docker.yaml --save_dir /root/log 