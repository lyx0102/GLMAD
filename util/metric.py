from functools import partial

from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
from skimage import measure
import multiprocessing
import copy

import numpy as np
from numba import jit
import torch
from torch import nn
from torch.nn import functional as F

from util.util import get_timepc, log_msg
from util.registry import Registry

from adeval import EvalAccumulatorCuda
import kornia

EVALUATOR = Registry('Evaluator')


def func(th, amaps, binary_amaps, masks):
    print('start', th)
    binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
    pro = []
    for binary_amap, mask in zip(binary_amaps, masks):
        for region in measure.regionprops(measure.label(mask)):
            tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
            pro.append(tp_pixels / region.area)
    inverse_masks = 1 - masks
    fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
    fpr = fp_pixels / inverse_masks.sum()
    print('end', th)
    return [np.array(pro).mean(), fpr, th]

def topk_score(pr_px, k=100):
    # pr_px: [N, H, W]
    N = pr_px.shape[0]
    pr_flat = pr_px.reshape(N, -1)
    topk_vals = np.partition(pr_flat, -k, axis=1)[:, -k:]
    return topk_vals.mean(axis=1)


def profile_multi_scale_image_score(
    pr_px,
    profiles,
    fusion="mean"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pr_px_tensor = torch.tensor(pr_px, dtype=torch.float32).unsqueeze(1).to(device)

    N = pr_px_tensor.shape[0]
    profile_scores = []

    for (k_size, stride, topk) in profiles:
        pad = k_size // 2

        pooled = F.avg_pool2d(
            pr_px_tensor,
            kernel_size=k_size,
            stride=stride,
            padding=pad
        )  # [N, 1, H', W']

        flat = pooled.flatten(1)  # [N, L]
        k = min(topk, flat.shape[1])

        topk_vals, _ = torch.topk(flat, k, dim=1)
        score = topk_vals.mean(dim=1)  # [N]

        profile_scores.append(score)

    profile_scores = torch.stack(profile_scores, dim=1)  # [N, num_profiles]

    if fusion == "mean":
        pr_sp = profile_scores.mean(dim=1)
    elif fusion == "max":
        pr_sp = profile_scores.max(dim=1)[0]
    else:
        raise ValueError(f"Unknown fusion method: {fusion}")

    return pr_sp.cpu().numpy()


class Evaluator(object):
    def __init__(self, metrics=[], pooling_ks=None, max_step_aupro=200, mp=False, use_adeval=False):
        if len(metrics) == 0:
            self.metrics = [
                'mAUROC_sp_max', 'mAUROC_px', 'mAUPRO_px',
                'mAP_sp_max', 'mAP_px',
                'mF1_max_sp_max',
                'mF1_px_0.2_0.8_0.1', 'mAcc_px_0.2_0.8_0.1', 'mIoU_px_0.2_0.8_0.1',
                'mF1_max_px', 'mIoU_max_px',
            ]
        else:
            self.metrics = metrics

        self.pooling_ks = pooling_ks
        self.max_step_aupro = max_step_aupro
        self.mp = mp

        self.eps = 1e-8
        self.beta = 1.0

        self.boundary = 1e-7
        self.use_adeval = use_adeval

    def run(self, results, cls_name, logger=None):
        idxes = results['cls_names'] == cls_name
        # print('idxes', idxes)
        # print('gt_px', results['imgs_masks'][idxes].shape)
        gt_px = results['imgs_masks'][idxes].squeeze(1)
        pr_px = results['anomaly_maps'][idxes]
        pr_sp = results['anomalys'][idxes]
        if 'smp_pre' in results:
            pr_sample = results['smp_pre'][idxes]
            subarrays = np.split(pr_sample, len(pr_sample)//5)
            pr_sa_max = np.array([np.max(subarray) for subarray in subarrays])
            subarray2 = np.split(idxes, len(idxes) // 5)
            sample_idxes = np.array([subarr[0] for subarr in subarray2])
            gt_sa = results['smp_masks'][sample_idxes]
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        # cls_names = results['cls_names'][idxes]
        # anomalys = results['anomalys'][idxes]
        # normalization for pixel-level evaluations
        pr_px_norm = (pr_px - pr_px.min()) / (pr_px.max() - pr_px.min())
        gt_sp = gt_px.max(axis=(1, 2))
        if self.pooling_ks is not None:
            # print("Using pooling & max")
            # pr_px_pooling = F.avg_pool2d(torch.tensor(pr_px).unsqueeze(1), self.pooling_ks, stride=1).numpy().squeeze(1)
            # pr_sp_max = pr_px_pooling.max(axis=(1, 2))
            # pr_sp_mean = pr_px_pooling.mean(axis=(1, 2))

            print("Using MSE-Scoring")
            CANDIDATE_PROFILES = [
                (8, 8, 50),
                (16, 1, 50),
                (16, 4, 100),
                (4, 1, 20),
                (8, 8, 100)
            ]
            pr_sp_max = profile_multi_scale_image_score(
                pr_px,
                profiles=CANDIDATE_PROFILES,
                fusion="mean"
            )


        # if self.pooling_ks == 'auto' or self.pooling_ks is not None:
        #     # 定义具体的配置方案 (Profile)，不再模糊搜索
        #     # 格式: (Kernel_Size, Stride, Top_K_Value)
        #     CANDIDATE_PROFILES = [
        #         # --- 方案 A: 针对 Model 1 类型 (关注区域，低分辨率) ---
        #         # Stride 8, 32x32=1024 pixels, K=50 (占比~5%)
        #         (8, 8, 50),
        #         # --- 方案 B: 针对 Model 2 类型 (关注尖峰，高分辨率，强平滑) ---
        #         # Stride 1, 256x256=65536 pixels, K=50 (占比~0.07%) -> 捕捉极值
        #         (16, 1, 50),
        #         # --- 方案 C: 折中方案 (如果异常稍微大一点) ---
        #         (16, 4, 100),
        #         # --- 方案 D: 极小异常捕捉 (配合较小 Kernel) ---
        #         (4, 1, 20),
        #         # --- 方案 E: 兜底方案 (类似全局平均) ---
        #         (8, 8, 100)
        #     ]
        #     # print(f"[{cls_name}] Auto-Search v2 running...")
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #     # 转换为 Tensor: [N, 1, H, W]
        #     pr_px_tensor = torch.tensor(pr_px, dtype=torch.float32).unsqueeze(1).to(device)
        #     gt_sp_np = gt_px.max(axis=(1, 2))
        #     best_auroc = -1.0
        #     best_scores = None
        #     best_profile = None
        #     for k_size, stride, k_val in CANDIDATE_PROFILES:
        #         # 1. 池化
        #         pad = k_size // 2
        #         # 这里的 padding 可能会导致边缘稍微有些 artifact，但影响通常很小
        #         pooled_map = F.avg_pool2d(pr_px_tensor, kernel_size=k_size, stride=stride, padding=pad)
        #         # 2. 展平
        #         N_b, _, H_b, W_b = pooled_map.shape
        #         flat_map = pooled_map.view(N_b, -1)
        #         # 3. 安全的 Top-K (防止 K 大于特征图像素数)
        #         current_k = min(k_val, flat_map.shape[1])
        #         # 4. 计算分数
        #         topk_values, _ = torch.topk(flat_map, current_k, dim=1)
        #         current_scores = topk_values.mean(dim=1).cpu().numpy()
        #         # 5. 计算 AUROC
        #         try:
        #             if len(np.unique(gt_sp_np)) < 2:
        #                 current_auroc = 0.0
        #             else:
        #                 current_auroc = roc_auc_score(gt_sp_np, current_scores)
        #         except:
        #             current_auroc = 0.0
        #         # print(f"   Profile [K:{k_size}, S:{stride}, TopK:{k_val}] -> AUROC: {current_auroc:.4f}")
        #         if current_auroc >= best_auroc:
        #             best_auroc = current_auroc
        #             best_scores = current_scores
        #             best_profile = (k_size, stride, k_val)
        #     # print(f"   >>> WINNER: {best_profile} with AUROC {best_auroc:.4f}")
        #     # 赋值给最终结果
        #     pr_sp_max = best_scores
        #     # Mean 分数由于计算逻辑不同，这里简单用最后一次池化的结果，或者你可以专门再算一次
        #     # 通常 sp_max 才是决定异常检测性能的关键
        #     pr_sp_mean = pr_px.mean(axis=(1, 2))

        else:
            print("Using max")
            pr_sp_max = pr_px.max(axis=(1, 2))
            # pr_sp_max = topk_score(pr_px, k=50)
            pr_sp_mean = pr_px.mean(axis=(1, 2))
            # pr_sp_max = topk_score(pr_px_sem, k=50)
            # pr_sp_mean = pr_px_sem.mean(axis=(1, 2))

        if self.use_adeval:
            score_min = min(pr_sp_max) - self.boundary
            score_max = max(pr_sp_max) + self.boundary
            anomap_min = pr_px.min()
            anomap_max = pr_px.max()
            accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max, skip_pixel_aupro=False, nstrips=50)
            accum.add_anomap_batch(torch.tensor(pr_px).cuda(non_blocking=True),
                                   torch.tensor(gt_px.astype(np.uint8)).cuda(non_blocking=True))
            for i in range(torch.tensor(pr_px).size(0)):
                accum.add_image(torch.tensor(pr_sp_max[i]), torch.tensor(gt_sp[i]))
            metrics = accum.summary()

        metric_str = f'==> Metric Time for {cls_name:<15}: '
        metric_results = {}
        for metric in self.metrics:
            t0 = get_timepc()
            if metric.startswith('mAUROC_sp_max'):
                auroc_sp = roc_auc_score(gt_sp, pr_sp_max)
                metric_results[metric] = auroc_sp
                # if not self.use_adeval:
                #     auroc_sp = roc_auc_score(gt_sp, pr_sp_max)
                #     metric_results[metric] = auroc_sp
                # else:
                #     metric_results[metric] = metrics['i_auroc']
            elif metric.startswith('mAUROC_sp_mean'):
                auroc_sp = roc_auc_score(gt_sp, pr_sp_mean)
                metric_results[metric] = auroc_sp
            elif metric.startswith('AUROC_sp'):
                auroc_sp = roc_auc_score(gt_sp, pr_sp)
                metric_results[metric] = auroc_sp
            elif metric.startswith('mAUROC_sa_max'):
                auroc_sp = roc_auc_score(gt_sa, pr_sa_max)
                metric_results[metric] = auroc_sp
            elif metric.startswith('mAUROC_px'):
                if not self.use_adeval:
                    auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
                    metric_results[metric] = auroc_px
                else:
                    metric_results[metric] = metrics['p_auroc']
            elif metric.startswith('mAUPRO_px'):
                if not self.use_adeval:
                    aupro_px = self.cal_pro_score(gt_px, pr_px, max_step=self.max_step_aupro, mp=self.mp)
                    metric_results[metric] = aupro_px
                else:
                    metric_results[metric] = metrics['p_aupro']
            elif metric.startswith('mAP_sp_max'):
                ap_sp = average_precision_score(gt_sp, pr_sp_max)
                metric_results[metric] = ap_sp
                # if not self.use_adeval:
                #     ap_sp = average_precision_score(gt_sp, pr_sp_max)
                #     metric_results[metric] = ap_sp
                # else:
                #     metric_results[metric] = metrics['i_aupr']
            elif metric.startswith('AP_sp_mean'):
                ap_sp = average_precision_score(gt_sp, pr_sp_mean)
                metric_results[metric] = ap_sp
            elif metric.startswith('mAP_px'):
                if not self.use_adeval:
                    ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
                    metric_results[metric] = ap_px
                else:
                    metric_results[metric] = metrics['p_aupr']
            elif metric.startswith('mAP_sa_max'):
                ap_sp = average_precision_score(gt_sa, pr_sa_max)
                metric_results[metric] = ap_sp
            elif metric.startswith('mF1_max_sp_max'):
                precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp_max)
                f1_scores = (2 * precisions * recalls) / (precisions + recalls)
                best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
                best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
                best_threshold = thresholds[best_f1_score_index]
                metric_results[metric] = best_f1_score
                # print('precisions', precisions)
                # print('recalls', recalls)
                # print('f1_scores', f1_scores)
                # print('thresholds', thresholds)
                # print(len(thresholds))
            elif metric.startswith('mF1_max_sa_max'):
                precisions, recalls, thresholds = precision_recall_curve(gt_sa, pr_sa_max)
                f1_scores = (2 * precisions * recalls) / (precisions + recalls)
                best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
                best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
                best_threshold = thresholds[best_f1_score_index]
                metric_results[metric] = best_f1_score
            elif metric.startswith('F1_max_sp_mean'):
                precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp_mean)
                f1_scores = (2 * precisions * recalls) / (precisions + recalls)
                best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
                best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
                best_threshold = thresholds[best_f1_score_index]
                metric_results[metric] = best_f1_score
            # elif metric.startswith('F1_max_px'):
            #     precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
            #     f1_scores = (2 * precisions * recalls) / (precisions + recalls)
            #     best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
            #     best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
            #     best_threshold = thresholds[best_f1_score_index]
            #     metric_results[metric] = best_f1_score
            elif metric.startswith('mF1_px') or metric.startswith('mDice_px') or metric.startswith('mAcc_px') or metric.startswith('mIoU_px'):  # example: F1_sp_0.3_0.8
                # F1_px equals Dice_px
                coms = metric.split('_')
                assert len(coms) == 5, f"{metric} should contain parameters 'score_l', 'score_h', and 'score_step' "
                score_l, score_h, score_step = float(metric.split('_')[-3]), float(metric.split('_')[-2]), float(metric.split('_')[-1])
                gt = gt_px.astype(np.bool_)
                metric_scores = []
                for score in np.arange(score_l, score_h + 1e-3, score_step):
                    pr = pr_px_norm > score
                    total_area_intersect = np.logical_and(gt, pr).sum(axis=(0, 1, 2))
                    total_area_union = np.logical_or(gt, pr).sum(axis=(0, 1, 2))
                    total_area_pred_label = pr.sum(axis=(0, 1, 2))
                    total_area_label = gt.sum(axis=(0, 1, 2))
                    if metric.startswith('mF1_px'):
                        precision = total_area_intersect / (total_area_pred_label + self.eps)
                        recall = total_area_intersect / (total_area_label + self.eps)
                        f1_px = (1 + self.beta ** 2) * precision * recall / (self.beta ** 2 * precision + recall + self.eps)
                        metric_scores.append(f1_px)
                    elif metric.startswith('mDice_px'):
                        dice_px = 2 * total_area_intersect / (total_area_pred_label + total_area_label + self.eps)
                        metric_scores.append(dice_px)
                    elif metric.startswith('mAcc_px'):
                        acc_px = total_area_intersect / (total_area_label + self.eps)
                        metric_scores.append(acc_px)
                    elif metric.startswith('mIoU_px'):
                        iou_px = total_area_intersect / (total_area_union + self.eps)
                        metric_scores.append(iou_px)
                    else:
                        raise f'invalid metric: {metric}'
                metric_results[metric] = np.array(metric_scores).mean()
            elif metric.startswith('mF1_max_px') or metric.startswith('mDice_max_px') or metric.startswith('mAcc_max_px') or metric.startswith('mIoU_max_px'):
                # F1_px equals Dice_px
                score_l, score_h, score_step = 0.0, 1.0, 0.05
                gt = gt_px.astype(np.bool_)
                metric_scores = []
                for score in np.arange(score_l, score_h + 1e-3, score_step):
                    pr = pr_px_norm > score
                    total_area_intersect = np.logical_and(gt, pr).sum(axis=(0, 1, 2))
                    total_area_union = np.logical_or(gt, pr).sum(axis=(0, 1, 2))
                    total_area_pred_label = pr.sum(axis=(0, 1, 2))
                    total_area_label = gt.sum(axis=(0, 1, 2))
                    if metric.startswith('mF1_max_px'):
                        precision = total_area_intersect / (total_area_pred_label + self.eps)
                        recall = total_area_intersect / (total_area_label + self.eps)
                        f1_px = (1 + self.beta ** 2) * precision * recall / (
                                    self.beta ** 2 * precision + recall + self.eps)
                        metric_scores.append(f1_px)
                    elif metric.startswith('mDice_max_px'):
                        dice_px = 2 * total_area_intersect / (total_area_pred_label + total_area_label + self.eps)
                        metric_scores.append(dice_px)
                    elif metric.startswith('mAcc_max_px'):
                        acc_px = total_area_intersect / (total_area_label + self.eps)
                        metric_scores.append(acc_px)
                    elif metric.startswith('mIoU_max_px'):
                        iou_px = total_area_intersect / (total_area_union + self.eps)
                        metric_scores.append(iou_px)
                    else:
                        raise f'invalid metric: {metric}'
                metric_results[metric] = np.array(metric_scores).max()
            t1 = get_timepc()
            metric_str += f'{t1 - t0:7.3f} ({metric})\t'
        log_msg(logger, metric_str)
        return metric_results

    @staticmethod
    def cal_anomaly_map(ft_list, fs_list, out_size=[224, 224], uni_am=False, use_cos='cos', amap_mode='add', gaussian_sigma=0, weights=None):
        msgms_metric = MSGMS_Score().to('cuda')
        ssim_metric = SSIM_Score(window_size=11).to("cuda")

        # ft_list = [f.cpu() for f in ft_list]
        # fs_list = [f.cpu() for f in fs_list]
        bs = ft_list[0].shape[0]
        weights = weights if weights else [1] * len(ft_list)
        anomaly_map = np.ones([bs] + out_size) if amap_mode == 'mul' else np.zeros([bs] + out_size)
        a_map_list = []
        if uni_am:
            size = (ft_list[0].shape[2], ft_list[0].shape[3])
            for i in range(len(ft_list)):
                ft_list[i] = F.interpolate(F.normalize(ft_list[i], p=2), size=size, mode='bilinear', align_corners=True)
                fs_list[i] = F.interpolate(F.normalize(fs_list[i], p=2), size=size, mode='bilinear', align_corners=True)
            ft_map, fs_map = torch.cat(ft_list, dim=1), torch.cat(fs_list, dim=1)
            if use_cos == 'cos':
                a_map = 1 - F.cosine_similarity(ft_map, fs_map, dim=1)
                a_map = a_map.unsqueeze(dim=1)
            elif use_cos == 'l2':
                a_map = torch.sqrt(torch.sum((ft_map - fs_map) ** 2, dim=1, keepdim=True))
            elif use_cos == 'msgms':
                a_map = msgms_metric(ft_map, fs_map)
            elif use_cos == 'ssim':
                a_map = ssim_metric(ft_map, fs_map)
            else:
                raise ValueError(f"Unknown metric: {use_cos}")
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
            a_map = a_map.squeeze(dim=1).cpu().detach().numpy()
            anomaly_map = a_map
            a_map_list.append(a_map)
        else:
            for i in range(len(ft_list)):
                ft = ft_list[i]
                fs = fs_list[i]
                # fs_norm = F.normalize(fs, p=2)
                # ft_norm = F.normalize(ft, p=2)
                if use_cos == 'cos':
                    a_map = 1 - F.cosine_similarity(ft, fs, dim=1)
                    a_map = a_map.unsqueeze(dim=1)
                elif use_cos == 'l2':
                    a_map = torch.sqrt(torch.sum((ft - fs) ** 2, dim=1, keepdim=True))
                elif use_cos == 'msgms':
                    a_map = msgms_metric(ft, fs)
                elif use_cos == 'ssim':
                    a_map = ssim_metric(ft, fs)
                else:
                    raise ValueError(f"Unknown metric: {use_cos}")
                a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
                a_map = a_map.squeeze(dim=1)
                a_map = a_map.cpu().detach().numpy()
                a_map_list.append(a_map)
                if amap_mode == 'add':
                    anomaly_map += a_map * weights[i]
                else:
                    anomaly_map *= a_map
            if amap_mode == 'add':
                anomaly_map /= (len(ft_list) * sum(weights))
        if gaussian_sigma > 0:
            for idx in range(anomaly_map.shape[0]):
                anomaly_map[idx] = gaussian_filter(anomaly_map[idx], sigma=gaussian_sigma)
        return anomaly_map, a_map_list

    @staticmethod
    def cal_pro_thr(results, th, amaps, masks):
        binary_amaps = np.zeros_like(amaps, dtype=np.bool_)
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        results.append([np.array(pro).mean(), fpr, th])
        # return [np.array(pro).mean(), fpr, th]

    @staticmethod
    def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3, mp=False):
        # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
        min_th, max_th = amaps.min(), amaps.max()
        delta = (max_th - min_th) / max_step
        pros, fprs, ths = [], [], []
        if mp:
            # enable in the main process, i.e., __main__
            # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count() * 0.25))
            # jobs = []
            # for th in np.arange(min_th, max_th, delta):
            #     job = pool.apply_async(Evaluator.cal_pro_thr, args=(th, amaps, masks, ))
            #     jobs.append(job)
            # pool.close()
            # pool.join()
            # results = []
            # for job in jobs:
            #     results.append(job.get())
            #     print(results[-1])

            # multiprocessing.Process has no `get` function
            results = multiprocessing.Manager().list()
            jobs = []
            for th in np.arange(min_th, max_th, delta):
                job = multiprocessing.Process(target=Evaluator.cal_pro_thr, args=(results, th, amaps, masks,))
                job.start()
                jobs.append(job)
            for job in jobs:
                job.join()
            results = list(results)
            results.sort(key=lambda x: float(x[2]))
            for result in results:
                pros.append(result[0])
                fprs.append(result[1])
                ths.append(result[2])
        else:
            binary_amaps = np.zeros_like(amaps, dtype=np.bool_)
            for th in np.arange(min_th, max_th, delta):
                binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
                pro = []
                for binary_amap, mask in zip(binary_amaps, masks):
                    for region in measure.regionprops(measure.label(mask)):
                        tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                        pro.append(tp_pixels / region.area)
                inverse_masks = 1 - masks
                fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
                fpr = fp_pixels / inverse_masks.sum()
                pros.append(np.array(pro).mean())
                fprs.append(fpr)
                ths.append(th)
        pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
        idxes = fprs < expect_fpr
        fprs = fprs[idxes]
        fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
        pro_auc = auc(fprs, pros[idxes])
        return pro_auc


def get_evaluator(cfg_evaluator):
    evaluator, kwargs = Evaluator, cfg_evaluator.kwargs
    return evaluator(**kwargs)

class Prewitt(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
        Gx = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]) / 3
        Gy = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]) / 3
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1).to('cuda')
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

def GMS(Ii, Ir, edge_filter, median_filter, c=0.0026):
    x = torch.mean(Ii, dim=1, keepdim=True)
    y = torch.mean(Ir, dim=1, keepdim=True)
    g_I = edge_filter(median_filter(x))
    g_Ir = edge_filter(median_filter(y))
    g_map = (2 * g_I * g_Ir + c) / (g_I**2 + g_Ir**2 + c)
    return g_map

class MSGMS_Score(nn.Module):
    def __init__(self):
        super().__init__()
        self.GMS = partial(GMS, edge_filter=Prewitt(), median_filter=kornia.filters.MedianBlur((3, 3)))
        self.median_filter = kornia.filters.MedianBlur((21, 21))

    def GMS_Score(self, Ii, Ir):
        return self.GMS(Ii, Ir)

    def forward(self, Ii, Ir):
        total_scores = self.GMS_Score(Ii, Ir)
        img_size = Ii.size(-1)
        total_scores = F.interpolate(total_scores, size=img_size, mode='bilinear', align_corners=False)
        for _ in range(3):
            Ii = F.avg_pool2d(Ii, kernel_size=2, stride=2)
            Ir = F.avg_pool2d(Ir, kernel_size=2, stride=2)
            score = self.GMS_Score(Ii, Ir)
            total_scores += F.interpolate(score, size=img_size, mode='bilinear', align_corners=False)
        # diff_map = 1 - (total_scores / 4.0)
        # return diff_map

        return (1 - total_scores) / 4

class SSIM_Score(nn.Module):
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, Ii, Ir):
        # 输入 [N,C,H,W]，必须 >= window_size
        ssim_map: torch.Tensor = kornia.metrics.ssim(Ii, Ir, window_size=self.window_size, reduction='none')
        # Kornia 输出 [N,1,H,W]，值越大越相似
        diff_map = 1 - ssim_map  # 越大越异常
        return diff_map
