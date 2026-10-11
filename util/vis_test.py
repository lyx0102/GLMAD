import numpy as np
import torch
import os
import matplotlib.cm as cm
import torch.nn as nn
import cv2
from PIL import Image
# import accimage
import torchvision
import torchvision.transforms as transforms
from skimage import color
import torch.nn.functional as F

def compute_iou(pred_mask, gt_mask):
    """计算 IoU（支持 numpy 数组）"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def find_best_iou_threshold(anomaly_map, gt_mask, thresholds=np.linspace(0, 1, 100)):
    """搜索最佳 IoU 所对应的阈值"""
    gt_mask = (gt_mask > 0).astype(np.bool_)
    best_iou = -1
    best_thresh = 0.0
    for t in thresholds:
        pred_mask = (anomaly_map >= t)
        iou = compute_iou(pred_mask, gt_mask)
        if iou > best_iou:
            best_iou = iou
            best_thresh = t
    return best_thresh, best_iou

def vis_rgb_gt_amp(img_paths, imgs, img_masks, anomaly_maps, method, root_out, dataset_name):
    if imgs.shape[-1] != img_masks.shape[-1]:
        imgs = F.interpolate(imgs, size=img_masks.shape[-1], mode='bilinear', align_corners=False)

    for idx, (img_path, img, img_mask, anomaly_map) in enumerate(zip(img_paths, imgs, img_masks, anomaly_maps)):
        parts = img_path.split('/')
        needed_parts = parts[1:-1]
        specific_root = '/'.join(needed_parts)
        img_num = parts[-1].split('.')[0]

        out_dir = f'{root_out}/{method}/{specific_root}'
        os.makedirs(out_dir, exist_ok=True)
        img_path_out = f'{out_dir}/{img_num}_img.png'
        img_ano_path = f'{out_dir}/{img_num}_amp.png'
        mask_path = f'{out_dir}/{img_num}_mask.png'

        # 恢复图像
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
        img_rec = img * std[:, None, None] + mean[:, None, None]
        img_rec = Image.fromarray((img_rec * 255).clamp(0, 255).type(torch.uint8).cpu().numpy().transpose(1, 2, 0))
        img_rec.save(img_path_out)

        # 归一化 anomaly map，增强对比度
        # vmin = np.percentile(anomaly_map, 5)
        # vmax = np.percentile(anomaly_map, 95)
        # anomaly_map = np.clip((anomaly_map - vmin) / (vmax - vmin + 1e-8), 0, 1)

        # anomaly_map = anomaly_map / anomaly_map.max()

        clip_max = 0.5  # 可以根据你数据经验调整（比如 0.3~0.7）
        anomaly_map = np.clip(anomaly_map, 0, clip_max)  # 截断最大值
        anomaly_map = anomaly_map / clip_max  # 归一化到0~1

        # anomaly_map = np.power(anomaly_map, 1.5)  # γ增强对比度
        # anomaly_map_unint8 = (anomaly_map * 255).astype(np.uint8)
        # kernel = np.ones((3, 3), np.uint8)
        # anomaly_map_erode = cv2.erode(anomaly_map_unint8, kernel, iterations=1)
        # anomaly_map = anomaly_map_erode.astype(np.float32) / 255.0


        # 生成热力图
        heatmap = cm.jet(anomaly_map)[..., :3]  # 去掉alpha通道
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap)

        # 融合热力图和原图
        img_rec_anomaly_map = Image.blend(img_rec, heatmap_img, alpha=0.4)
        img_rec_anomaly_map.save(img_ano_path)

        # 保存 mask
        img_mask_np = (img_mask * 255).astype(np.uint8).transpose(1, 2, 0)
        if img_mask_np.shape[2] == 1:
            img_mask_np = np.repeat(img_mask_np, 3, axis=2)
        mask_img = Image.fromarray(img_mask_np)
        mask_img.save(mask_path)

# def vis_rgb_gt_amp(
#     img_paths, imgs, img_masks, anomaly_maps,
#     method, root_out, dataset_name
# ):
#     if imgs.shape[-1] != img_masks.shape[-1]:
#         imgs = F.interpolate(imgs, size=img_masks.shape[-1], mode='bilinear', align_corners=False)
#
#     mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device)
#     std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device)
#
#     iou_list = []
#
#     for idx, (img_path, img, img_mask, anomaly_map) in enumerate(zip(img_paths, imgs, img_masks, anomaly_maps)):
#         parts = img_path.split('/')
#         specific_root = '/'.join(parts[1:-1])
#         img_num = parts[-1].split('.')[0]
#         out_dir = os.path.join(root_out, method, specific_root)
#         os.makedirs(out_dir, exist_ok=True)
#
#         # 文件名
#         img_path_save  = os.path.join(out_dir, f'{img_num}_img.png')
#         amp_path_save  = os.path.join(out_dir, f'{img_num}_amp.png')
#         mask_path_save = os.path.join(out_dir, f'{img_num}_mask.png')
#         pred_mask_save = os.path.join(out_dir, f'{img_num}_pred_mask.png')
#
#         # 原图反标准化
#         img_rec = img * std[:, None, None] + mean[:, None, None]
#         img_rec = (img_rec * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
#         img_pil = Image.fromarray(img_rec)
#         img_pil.save(img_path_save)
#
#         # anomaly_map: Tensor or np.ndarray
#         if isinstance(anomaly_map, torch.Tensor):
#             amap = anomaly_map.detach().cpu().numpy()
#         else:
#             amap = anomaly_map
#         amap = amap / (amap.max() + 1e-8)
#
#         # ------------------- 搜索最佳 IoU 阈值 -------------------
#         gt_mask = img_mask.squeeze() if img_mask.ndim == 3 else img_mask
#         if isinstance(gt_mask, torch.Tensor):
#             gt_mask = gt_mask.detach().cpu().numpy()
#         gt_mask = (gt_mask > 0).astype(np.uint8)
#
#         best_thresh, best_iou = find_best_iou_threshold(amap, gt_mask)
#         iou_list.append(best_iou)
#         # print(f"[{img_num}] Best IoU: {best_iou:.4f}, Threshold: {best_thresh:.2f}")
#
#         # ------------------- 可视化热图 -------------------
#         cmap = cm.jet(amap)[:, :, :3] * 255
#         cmap = cmap.astype(np.uint8)
#         cmap_pil = Image.fromarray(cmap)
#         blend = Image.blend(img_pil, cmap_pil, alpha=0.4)
#         blend.save(amp_path_save)
#
#         # ------------------- 保存 GT mask -------------------
#         gt_rgb = np.stack([gt_mask * 255] * 3, axis=2).astype(np.uint8)
#         Image.fromarray(gt_rgb).save(mask_path_save)
#
#         # ------------------- 保存预测 mask -------------------
#         pred_mask = (amap >= best_thresh).astype(np.uint8) * 255
#         pred_rgb = np.stack([pred_mask] * 3, axis=2)
#         Image.fromarray(pred_rgb).save(pred_mask_save)

    # # 可选：输出平均 IoU
    # avg_iou = sum(iou_list) / len(iou_list)
    # print(f"\n📊 平均最佳 IoU: {avg_iou:.4f}")