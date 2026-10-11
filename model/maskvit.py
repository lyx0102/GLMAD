import random

import torch
import time
import numpy as np
from torch import nn

from model import get_model
from model import MODEL
from functools import partial
from timm.models.vision_transformer import VisionTransformer, checkpoint_filter_fn
from timm.models.helpers import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, RmsNorm


# ============================== Fusion ==============================
class ChannelAttention(nn.Module):
	def __init__(self, in_channels):
		super(ChannelAttention, self).__init__()
		self.global_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
			nn.Sigmoid()
		)

	def forward(self, x):
		w = self.global_pool(x)
		w = self.fc(w)
		return x * w

class Fusion(nn.Module):
	def __init__(self):
		super(Fusion, self).__init__()

		self.fuse = nn.Sequential(
			nn.Conv2d(1536, 384, kernel_size=1),
			nn.BatchNorm2d(384),
			nn.ReLU(inplace=True)
		)
		self.att = ChannelAttention(384)

	def forward(self, features):
		# B, L, C
		B, L, C = features[0].shape
		H = int(np.sqrt(L))
		feature_align = []
		for feature in features:
			feature_align.append(feature.view(B, H, H, C).permute(0, 3, 1, 2).contiguous())

		feature_align = torch.cat(feature_align, dim=1)
		feature_align = self.fuse(feature_align)
		feature_align = self.att(feature_align)
		feature_align = feature_align.permute(0, 2, 3, 1).contiguous().view(B, -1, C)

		return feature_align

	# # only Add
	# def forward(self, features):
	# 	# B, L, C
	# 	feature_align = torch.zeros_like(features[0])
	# 	for i in range(len(features)):
	# 		feature_align += features[i]
	#
	# 	return feature_align

@MODEL.register_module
def fusion_parallel_re(pretrained=False, **kwargs):
	model = Fusion(**kwargs)
	return model


# ============================== Encoder ==============================
class ViT_Encoder(VisionTransformer):
	def __init__(self, teachers, neck, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.teachers = teachers
		self.neck = neck

	def forward(self, x):
		x = self.patch_embed(x)
		x = self._pos_embed(x)

		out_neck, out_t = [], []
		if self.grad_checkpointing and not torch.jit.is_scripting():
			x = checkpoint_seq(self.blocks, x)
		else:
			for i in range(len(self.blocks)):
				x = self.blocks[i](x)
				fea = x[:, 1:, :]
				if i+1 in self.neck:
					out_neck.append(fea)
				if i+1 in self.teachers:
					B, L, C = fea.shape
					H = int(np.sqrt(L))
					fea = fea.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
					out_t.append(fea)

		return out_t, out_neck

# ========== Encoders with Different Pre-trained Weights ==============
def create_vision_transformer(variant, pretrained=False, **kwargs):
	if kwargs.get('features_only', None):
		raise RuntimeError('features_only not implemented for Vision Transformer models.')

	if 'flexi' in variant:
		_filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
	else:
		_filter_fn = checkpoint_filter_fn
	finish = False
	while not finish:
		try:
			model = build_model_with_cfg(ViT_Encoder, variant, pretrained, pretrained_filter_fn=_filter_fn, **kwargs)
			finish = True
		except:
			print('Try load model for MyAD')
			time.sleep(1)
	return model

@MODEL.register_module
def _vit_small_patch16_224_dino(pretrained=True, **kwargs):
	""" ViT-Small-DINO, patch16"""
	model_kwargs = dict(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=384, depth=12, num_heads=6)
	model = create_vision_transformer('vit_small_patch16_224.dino', pretrained=pretrained, **dict(model_kwargs, **kwargs))
	return model

# ============================== Decoder ==============================
class Attention(nn.Module):
	def __init__(
			self,
			dim,
			num_heads=8,
			qkv_bias=False,
			qk_norm=False,
			attn_drop=0.,
			proj_drop=0.,
			norm_layer=nn.LayerNorm,
	):
		super().__init__()
		assert dim % num_heads == 0, 'dim should be divisible by num_heads'
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.scale = self.head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
		self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x, attn_mask=None):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
		q, k, v = qkv.unbind(0)
		q, k = self.q_norm(q), self.k_norm(k)

		q = q * self.scale
		attn = q @ k.transpose(-2, -1)

		if attn_mask is not None:
			attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)

		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		x = attn @ v

		x = x.transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x

class LayerScale(nn.Module):
	def __init__(self, dim, init_values=1e-5, inplace=False):
		super().__init__()
		self.inplace = inplace
		self.gamma = nn.Parameter(init_values * torch.ones(dim))

	def forward(self, x):
		return x.mul_(self.gamma) if self.inplace else x * self.gamma

class TransformerBlock(nn.Module):
	def __init__(
			self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False, drop=0., attn_drop=0., init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.attn = Attention(
			dim,
			num_heads=num_heads,
			qkv_bias=qkv_bias,
			qk_norm=qk_norm,
			attn_drop=attn_drop,
			proj_drop=drop,
			norm_layer=norm_layer,
		)
		self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
		self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

		self.norm2 = norm_layer(dim)
		self.mlp = Mlp(
			in_features=dim,
			hidden_features=int(dim * mlp_ratio),
			act_layer=act_layer,
			drop=drop,
		)
		self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
		self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

	def forward(self, x, attn_mask=None):
		# print('TransformerBlock attn_mask shape: ', attn_mask.shape)
		x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
		x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
		return x

class DecoderLayer(nn.Module):
	def __init__(self, dim, num_heads, mask='None'):
		super(DecoderLayer, self).__init__()
		self.block1 = TransformerBlock(dim=dim, num_heads=num_heads)
		self.block2 = TransformerBlock(dim=dim, num_heads=num_heads)
		self.block3 = TransformerBlock(dim=dim, num_heads=num_heads)
		# self.block4 = TransformerBlock(dim=dim, num_heads=num_heads)
		self.mask = mask

	def generate_rand_mask(self, feature_size, mask_ratio=0.3, device="cuda"):
		H, W = feature_size
		N = H * W

		mask = torch.zeros(N, N)

		total_elements = N * N
		num_mask = int(total_elements * mask_ratio)

		idx = torch.randperm(total_elements)[:num_mask]
		mask.view(-1)[idx] = float('-inf')

		return mask.to(device)

	def generate_neighbor_mask(self, feature_size, neighbor_size, device="cuda"):
		h, w = feature_size
		hm, wm = neighbor_size
		mask = torch.ones(h, w, h, w)
		for idx_h1 in range(h):
			for idx_w1 in range(w):
				idx_h2_start = max(idx_h1 - hm // 2, 0)
				idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
				idx_w2_start = max(idx_w1 - wm // 2, 0)
				idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
				mask[idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end] = 0
		mask = mask.view(h * w, h * w)
		mask = (
			mask.float()
			.masked_fill(mask == 0, float("-inf"))
			.masked_fill(mask == 1, float(0.0))
			.cuda()
		)
		return mask.to(device)

	def forward(self, x):
		# x: [B, L, C]
		B, L, C = x.shape
		H = int(np.sqrt(L))

		mask = None
		if self.mask == 'rand':
			mask_1 = self.generate_rand_mask(feature_size=[H, H], mask_ratio=0.3, device=x.device)
			mask_2 = self.generate_rand_mask(feature_size=[H, H], mask_ratio=0.3, device=x.device)
			mask_3 = self.generate_rand_mask(feature_size=[H, H], mask_ratio=0.3, device=x.device)
			# print('mask_1 shape: ', mask_1.shape)
			out = self.block1(x, mask_1)
			out = self.block2(out, mask_2)
			out = self.block3(out, mask_3)
		elif self.mask == 'neighbor':
			mask = self.generate_neighbor_mask(feature_size=[H, H], neighbor_size=(8, 8), device=x.device)
			out = self.block1(x, mask)
			out = self.block2(out, mask)
			out = self.block3(out, mask)
		elif self.mask == 'None':
			mask = None
			out = self.block1(x,mask)
			out = self.block2(out,mask)
			out = self.block3(out, mask)

		return out

class Decoder(nn.Module):
	def __init__(self, dim, num_heads, img_size=256, patch_size=16, in_chans=3, embed_dim=384, drop_rate=0., pre_norm=False):
		super(Decoder, self).__init__()
		self.patch_embed = PatchEmbed(
			img_size=img_size,
			patch_size=patch_size,
			in_chans=in_chans,
			embed_dim=embed_dim,
			bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
		)
		num_patches = self.patch_embed.num_patches
		# num_patches = 256
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
		trunc_normal_(self.pos_embed, std=.02)
		self.pos_drop = nn.Dropout(p=drop_rate)

		self.layer_1 = DecoderLayer(dim=dim, num_heads=num_heads, mask='neighbor')
		self.layer_2 = DecoderLayer(dim=dim, num_heads=num_heads, mask='rand')
		self.layer_3 = DecoderLayer(dim=dim, num_heads=num_heads, mask='rand')

	def add_jitter(self, feature_tokens, scale, prob):
		# feature_tokens 的维度是 [B, L, C]
		if random.uniform(0, 1) <= prob:
			batch_size, num_tokens, dim_channel = feature_tokens.shape  # 修改维度顺序
			feature_norms = (
					feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
			)  # [B, L, 1] 计算每个时间步/空间步的特征范数并归一化
			jitter = torch.randn((batch_size, num_tokens, dim_channel)).cuda()  # 修改形状
			jitter = jitter * feature_norms * scale  # 将噪声按特征范数缩放
			feature_tokens = feature_tokens + jitter  # 在原特征上添加噪声
		return feature_tokens

	def forward(self, x):
		# x: [B, L, C]
		B, L, C = x.shape
		H = int(np.sqrt(L))

		if self.training:
			x = self.add_jitter(feature_tokens=x, scale=20.0, prob=1.0)
		# x = self.patch_embed(x)
		x = x + self.pos_embed
		x = self.pos_drop(x)

		out = []
		layer_out = self.layer_1(x)
		_layer_out = layer_out.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
		out.append(_layer_out)

		layer_out = self.layer_2(layer_out)
		_layer_out = layer_out.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
		out.append(_layer_out)

		layer_out = self.layer_3(layer_out)
		_layer_out = layer_out.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
		out.append(_layer_out)

		return [out[int(len(out)-1-i)] for i in range(len(out))]

@MODEL.register_module
def create_decoder_parallel(pretrained=False, **kwargs):
	model = Decoder(**kwargs)
	return model

class Parallel_Re(torch.nn.Module):
	def __init__(self, model_backbone, model_fusion, model_decoder):
		super(Parallel_Re, self).__init__()
		self.model_backbone = get_model(model_backbone)
		self.model_fusion = get_model(model_fusion)
		self.model_decoder = get_model(model_decoder)

		self.frozen_layers = ['model_backbone']

	def freeze_layer(self, module):
		module.eval()
		for param in module.parameters():
			param.requires_grad = False

	def train(self, mode=True):
		self.training = mode
		for mname, module in self.named_children():
			if mname in self.frozen_layers:
				self.freeze_layer(module)
			else:
				module.train(mode)
		return self

	def forward(self, imgs):
		'''

		Args:
			imgs: [batch_size, channel, H, W]

		Returns:
			feats_t: list, len=3, [batch_size, channel, H, W]
			feats_n: list, len=1, [batch_size, channel, H, W]
		'''
		feats_t, feats_n = self.model_backbone(imgs)
		feats_t = [f.detach() for f in feats_t]
		feats_n = [f.detach() for f in feats_n]
		feats_n = self.model_fusion(feats_n)
		feats_s = self.model_decoder(feats_n)

		return feats_t, feats_s

@MODEL.register_module
def parallel_re(pretrained=False, **kwargs):
	model = Parallel_Re(**kwargs)
	return model

if __name__ == '__main__':
	from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
	from util.util import get_timepc, get_net_params
	from argparse import Namespace as _Namespace

	bs = 2
	reso = 256
	x = torch.randn(bs, 3, reso, reso).cuda()

	model_backbone = _Namespace()
	model_backbone.name = '_vit_small_patch16_224_dino'
	model_backbone.kwargs = dict(pretrained=True, checkpoint_path='', strict=True, img_size=reso, teachers=[3, 6, 9], neck=[3, 6, 9, 12])

	model_fusion = _Namespace()
	model_fusion.name = 'fusion_parallel_re'
	model_fusion.kwargs = dict(pretrained=False, checkpoint_path='', strict=False)

	model_decoder = _Namespace()
	model_decoder.name = 'create_decoder_parallel'
	model_decoder.kwargs = dict(pretrained=False, checkpoint_path='', strict=False, dim=384, num_heads=6, img_size=reso)

	model = parallel_re(model_backbone=model_backbone, model_fusion=model_fusion, model_decoder=model_decoder)
	# 保存模型结构
	# torch.save(model, 'myad.pth')
	model = model.cuda()
	feats_t, feats_s = model(x)
	print(feats_t[0].shape, feats_t[1].shape, feats_t[2].shape)
	print(feats_s[0].shape, feats_s[1].shape, feats_s[2].shape)