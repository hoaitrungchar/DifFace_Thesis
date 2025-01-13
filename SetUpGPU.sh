export CUDA_VISIBLE_DEVICES="1,3,4,5,6"
torchrun --standalone --nproc_per_node=1 --nnodes=1 main.py --cfg_path configs/training/predicted_edge_SwinUnet.yaml --save_dir /data/FFHQ/DifFace_prior