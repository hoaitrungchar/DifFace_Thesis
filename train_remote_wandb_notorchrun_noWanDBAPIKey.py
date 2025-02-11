from modal import App,Image,gpu,Mount,Volume
import modal 
model_name = "FFHQ_predicted_mask_SwinUnet"
app = App(f'Training {model_name} with H100 batch 256')

image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")\
        .apt_install(["ffmpeg","libsm6","libxext6",'libgl1'])\
        .pip_install("einops")\
        .pip_install("tensorboard")\
        .pip_install("opencv-python")\
        .pip_install("timm")\
        .pip_install("wandb")\
        .pip_install("omegaconf")\
        .pip_install("lpips")\
        .pip_install(['loguru','scikit-learn', 'scikit-image'])
        

@app.function(
    image=image,
    gpu = "a100-80gb",
    # gpu="T4",
    timeout=3600*24,
    mounts = [
        Mount.from_local_dir("basicsr",remote_path="/root/basicsr"),
        Mount.from_local_dir("configs",remote_path="/root/configs"),
        Mount.from_local_dir("datapipe",remote_path="/root/datapipe"),
        Mount.from_local_dir("facelib",remote_path="/root/facelib"),
        Mount.from_local_dir("ckpts",remote_path="/root/source/ckpts"),
        Mount.from_local_dir("ResizeRight",remote_path="/root/ResizeRight"),
        Mount.from_local_dir("scripts",remote_path="/root/scripts"),
        Mount.from_local_dir("utils",remote_path="/root/utils"),
        
        Mount.from_local_file("main.py","/root/main.py"),
        Mount.from_local_file("trainer.py","/root/trainer.py"),
        Mount.from_local_file("model_38500.pth","/root/model_38500.pth"),

    ],
    volumes={
        "/root/FFHQ_mask_log": Volume.from_name("FFHQ_mask_log"),
        "/root/data/FFHQ/train": Volume.from_name("FFHQ_train"),
        "/root/data/FFHQ/val": Volume.from_name("FFHQ_val"),
        "/root/data/CelebA-HQ/train": Volume.from_name("CelebAHQ_train"),
        "/root/data/CelebA-HQ/val": Volume.from_name("CelebAHQ_val"),
        "/root/data/ImageNet/train1": Volume.from_name("ImageNet_train1"),
        "/root/data/ImageNet/train2": Volume.from_name("ImageNet_train2"),
        "/root/data/ImageNet/train3": Volume.from_name("ImageNet_train3"),
        "/root/data/ImageNet/train4": Volume.from_name("ImageNet_train4"),
        "/root/data/ImageNet/train5": Volume.from_name("ImageNet_train5"),
        "/root/data/ImageNet/train6": Volume.from_name("ImageNet_train6"),
        "/root/data/ImageNet/train7": Volume.from_name("ImageNet_train7"),
        "/root/data/ImageNet/train8": Volume.from_name("ImageNet_train8"),
        "/root/data/ImageNet/train9": Volume.from_name("ImageNet_train9"),
        "/root/data/ImageNet/train10": Volume.from_name("ImageNet_train10"),
        "/root/data/ImageNet/train11": Volume.from_name("ImageNet_train11"),
        "/root/data/ImageNet/train12": Volume.from_name("ImageNet_train12"),
        "/root/data/ImageNet/train13": Volume.from_name("ImageNet_train13"),
        "/root/data/ImageNet/val": Volume.from_name("ImageNet_val"),
        "/root/data/Mask/train": Volume.from_name("Mask_train1"),
        "/root/data/Mask/val": Volume.from_name("Mask_val")
        

    }
)
def main():
    import subprocess
    import os 
    from omegaconf import OmegaConf
    from utils.util_common import get_obj_from_str

    os.environ['WANDB_API_KEY'] ="bfac2552df6b0b2f6c2aa6de31e4a16107dac2ac"
    cfg_path = "configs/training/predicted_mask_SwinUnet.yaml"
    configs = OmegaConf.load(cfg_path)

    # merge args to config
    configs['save_dir'] = 'FFHQ_mask_log'
    configs['resume'] = "/root/model_38500.pth"
    configs['seed'] = 1000

    trainer = get_obj_from_str(configs.trainer.target)(configs)
    trainer.train()
