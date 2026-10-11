# GLM-AD

Official code implementation of the paper "GLM-AD: A Dual-Encoder Framework with Global-to-Local Modulation for Multi-Class Unsupervised Anomaly Detection"


CUDA_VISIBLE_DEVICES=0 python run.py -c configs/HybridCaT/hybridcat_mvtec.py -m train

CUDA_VISIBLE_DEVICES=0 python run.py -c configs/MaskViT/maskvit_mvtec.py -m train

CUDA_VISIBLE_DEVICES=0 python run.py -c configs/Parallel_Reconstruction/parallel_reconstruction_mvtec.py -m train

CUDA_VISIBLE_DEVICES=0 python run.py -c configs/InpaintingNet/inpainting_mvtec.py -m train

CUDA_VISIBLE_DEVICES=0 python run.py -c configs/GLMAD/glmad_mvtec.py -m train

CUDA_VISIBLE_DEVICES=0 python run.py -c configs/GLMAD/glmad_visa.py -m train
