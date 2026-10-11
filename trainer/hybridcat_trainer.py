import gc
import os
import copy
import glob
import shutil
import datetime
import time

import tabulate
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator
from timm.data import Mixup

import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
    from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad

from ._base_trainer import BaseTrainer
from . import TRAINER
from util.vis import vis_rgb_gt_amp

@TRAINER.register_module
class HybridCaT_Trainer(BaseTrainer):
    def __init__(self, cfg):
        super(HybridCaT_Trainer, self).__init__(cfg)

    def set_input(self, inputs):
        self.imgs = inputs['img'].cuda()
        # print(self.imgs.shape)
        # print(self.imgs.min().item(), self.imgs.max().item())
        self.imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']
        self.bs = self.imgs.shape[0]

    def forward(self):
        self.feats_t, self.feats_s = self.net(self.imgs)
        # self.feats_t, self.feats_s, self.intermediates_all = self.net(self.imgs)

        # batch_size = self.imgs.shape[0]
        #
        # layer_3 = self.intermediates_all[2]
        #
        # layer_3_att_1 = layer_3['out_att'][0]
        # layer_3_att_2 = layer_3['out_att'][1]
        # layer_3_att_3 = layer_3['out_att'][2]
        # layer_3_y1 = layer_3['y1']
        # layer_3_y2 = layer_3['y2']
        # layer_3_y3 = layer_3['y3']
        # layer_3_y4 = layer_3['y4']
        #
        # for i in range(batch_size):
        #     base_dir = 'CKA'
        #     old_path = self.img_path[i]
        #     relative_path = old_path.replace('data/', '')
        #     relative_path = os.path.splitext(relative_path)[0]
        #
        #     # layer_3
        #     layer_3_att_1_path = os.path.join(base_dir, 'layer_3_att_1', f"{relative_path}.pt")
        #     os.makedirs(os.path.dirname(layer_3_att_1_path), exist_ok=True)
        #     torch.save(layer_3_att_1[i], layer_3_att_1_path)
        #     layer_3_att_2_path = os.path.join(base_dir, 'layer_3_att_2', f"{relative_path}.pt")
        #     os.makedirs(os.path.dirname(layer_3_att_2_path), exist_ok=True)
        #     torch.save(layer_3_att_2[i], layer_3_att_2_path)
        #     layer_3_att_3_path = os.path.join(base_dir, 'layer_3_att_3', f"{relative_path}.pt")
        #     os.makedirs(os.path.dirname(layer_3_att_3_path), exist_ok=True)
        #     torch.save(layer_3_att_3[i], layer_3_att_3_path)
        #
        #     layer_3_y1_path = os.path.join(base_dir, 'layer_3_y1', f"{relative_path}.pt")
        #     os.makedirs(os.path.dirname(layer_3_y1_path), exist_ok=True)
        #     torch.save(layer_3_y1[i], layer_3_y1_path)
        #     layer_3_y2_path = os.path.join(base_dir, 'layer_3_y2', f"{relative_path}.pt")
        #     os.makedirs(os.path.dirname(layer_3_y2_path), exist_ok=True)
        #     torch.save(layer_3_y2[i], layer_3_y2_path)
        #     layer_3_y3_path = os.path.join(base_dir, 'layer_3_y3', f"{relative_path}.pt")
        #     os.makedirs(os.path.dirname(layer_3_y3_path), exist_ok=True)
        #     torch.save(layer_3_y3[i], layer_3_y3_path)
        #     layer_3_y4_path = os.path.join(base_dir, 'layer_3_y4', f"{relative_path}.pt")
        #     os.makedirs(os.path.dirname(layer_3_y4_path), exist_ok=True)
        #     torch.save(layer_3_y4[i], layer_3_y4_path)

        # last_weights_avg = self.net.model_fusion.last_weights_avg
        # last_weights_full = self.net.model_fusion.last_weights_full
        # for i, avg in enumerate(last_weights_avg):
        #     full = last_weights_full[i]     # avg: [4], full: [384,4]
        #     old_path = self.img_path[i]
        #     # 构造新路径：基础路径 + 相对路径
        #     base_dir = 'visual_analysis/AWMF_visual_analysis'
        #     relative_path = old_path.replace('data/', '')  # 去掉前缀
        #     relative_path = os.path.splitext(relative_path)[0]  # 去掉.png加上.np
        #     # 构造 avg 路径
        #     avg_path = os.path.join(base_dir, 'avg', relative_path)
        #     os.makedirs(os.path.dirname(avg_path), exist_ok=True)
        #     np.save(avg_path, avg)  # 保存 avg 数组
        #     # 构造 full 路径
        #     full_path = os.path.join(base_dir, 'full', relative_path)
        #     os.makedirs(os.path.dirname(full_path), exist_ok=True)
        #     np.save(full_path, full)  # 保存 full 数组

    def optimize_parameters(self):
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        with self.amp_autocast():
            self.forward()
            loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s)
        self.backward_term(loss_cos, self.optim)
        update_log_term(self.log_terms.get('cos'), reduce_tensor(loss_cos, self.world_size).clone().detach().item(), 1,self.master)

    @torch.no_grad()
    def test(self):
        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
            if os.path.exists(f'{self.tmp_dir}/batch_result'):
                shutil.rmtree(f'{self.tmp_dir}/batch_result')
            os.makedirs(f'{self.tmp_dir}/batch_result', exist_ok=True)

        self.reset(isTrain=False)
        test_length = self.cfg.data.test_size
        # test_loader = iter(self.test_loader)

        for cls_name in self.cls_names:
            cls_number = 0
            batch_idx = 0
            print(cls_name, '开始测试')
            imgs_masks, anomaly_maps, cls_names, anomalys = [], [], [], []
            test_loader = iter(self.test_loader)
            while batch_idx < test_length:
                t1 = get_timepc()
                batch_idx += 1
                test_data = next(test_loader)
                self.set_input(test_data)
                self.forward()
                loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s)
                update_log_term(self.log_terms.get('cos'), reduce_tensor(loss_cos, self.world_size).clone().detach().item(),1, self.master)
                # get anomaly maps
                anomaly_map, _ = self.evaluator.cal_anomaly_map(self.feats_t, self.feats_s,
                                                                [self.imgs.shape[2], self.imgs.shape[3]], uni_am=False,
                                                                amap_mode='add', gaussian_sigma=4)
                # print(anomaly_map.shape)
                self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
                # print(self.imgs_mask.shape)[8, 1, 256, 256]
                if self.cfg.vis:
                    if self.cfg.vis_dir is not None:
                        root_out = self.cfg.vis_dir
                    else:
                        root_out = self.writer.logdir
                    vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int), anomaly_map,
                                   self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1])

                for i, name in enumerate(self.cls_name):
                    if name == cls_name:
                        cls_number = cls_number + 1
                        imgs_masks.append(self.imgs_mask[i].unsqueeze(0).cpu().numpy().astype(int))
                        anomaly_maps.append(np.expand_dims(anomaly_map[i],axis=0))
                        cls_names.append(np.array([cls_name]))
                        anomalys.append(np.array([self.anomaly[i].cpu().numpy().astype(int)]))

                t2 = get_timepc()
                update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
                print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
                # ---------- log ----------
                if self.master:
                    if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                        msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
                        log_msg(self.logger, msg)

            print(cls_name, '共有',cls_number)
            imgs_masks = np.concatenate(imgs_masks, axis=0)
            anomaly_maps = np.concatenate(anomaly_maps, axis=0)
            cls_names = np.concatenate(cls_names, axis=0)
            anomalys = np.concatenate(anomalys, axis=0)
            cls_result = {
                'imgs_masks': imgs_masks,
                'anomaly_maps': anomaly_maps,
                'cls_names': cls_names,
                'anomalys': anomalys
            }
            torch.save(cls_result, f'{self.tmp_dir}/{cls_name}.pt')
            del imgs_masks, anomaly_maps, cls_names, anomalys

        # merge results
        if self.cfg.dist:
            results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
            torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
            if self.master:
                results = dict(imgs_masks=[], anomaly_maps=[], cls_names=[], anomalys=[])
                valid_results = False
                while not valid_results:
                    results_files = glob.glob(f'{self.tmp_dir}/*.pth')
                    if len(results_files) != self.cfg.world_size:
                        time.sleep(1)
                    else:
                        idx_result = 0
                        while idx_result < self.cfg.world_size:
                            results_file = results_files[idx_result]
                            try:
                                result = torch.load(results_file)
                                for k, v in result.items():
                                    results[k].extend(v)
                                idx_result += 1
                            except:
                                time.sleep(1)
                        valid_results = True
        else:
            # results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
        # if self.master:
        #     results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
            msg = {}
            for idx, cls_name in enumerate(self.cls_names):
                results = torch.load(f'{self.tmp_dir}/{cls_name}.pt', weights_only=False)
                # results = torch.load(f'runs/tmp/{cls_name}.pt')

                print(cls_name, '加载完成，开始计算', results['cls_names'].shape)

                metric_results = self.evaluator.run(results, cls_name, self.logger)
                msg['Name'] = msg.get('Name', [])
                msg['Name'].append(cls_name)
                avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
                msg['Name'].append('Avg') if avg_act else None
                # msg += f'\n{cls_name:<10}'
                for metric in self.metrics:
                    metric_result = metric_results[metric] * 100
                    self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
                    max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
                    max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
                    msg[metric] = msg.get(metric, [])
                    msg[metric].append(metric_result)
                    msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
                    msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
                    if avg_act:
                        metric_result_avg = sum(msg[metric]) / len(msg[metric])
                        self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
                        max_metric = max(self.metric_recorder[f'{metric}_Avg'])
                        max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
                        msg[metric].append(metric_result_avg)
                        msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
                # 释放不再需要的内存
                # del imgs_maskss, anomaly_mapss, cls_namess, anomalyss, results
                del results
                gc.collect()

            msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center",
                                    stralign="center", )
            log_msg(self.logger, f'\n{msg}')
            shutil.rmtree(f'{self.tmp_dir}/batch_result')
    def visual_analysis(self):
        print('visual_analysis')