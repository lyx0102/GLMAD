import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
try:
	from torch.hub import load_state_dict_from_url
except ImportError:
	from torch.utils.model_zoo import load_url as load_state_dict_from_url
from timm.models.vision_transformer import VisionTransformer, _cfg, checkpoint_filter_fn
from timm.models.helpers import build_model_with_cfg
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from functools import partial
from model import get_model
from model import MODEL
from torchvision import transforms
import model.vision_transformer as vits
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, RmsNorm

'''========== Fusion =========='''
class GlobalFiLM(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(dim, dim),
			nn.ReLU(inplace=True),
			nn.Linear(dim, dim * 2)
		)

	def forward(self, f_local, f_global):
		"""
		f_local:  [B, L, C]   (L = 32*32)
		f_global: [B, C, 16, 16]
		"""
		# Global pooling
		g = f_global.mean(dim=[2, 3])     # [B, C]

		gamma, beta = self.mlp(g).chunk(2, dim=1)
		gamma = gamma.unsqueeze(1)        # [B, 1, C]
		beta = beta.unsqueeze(1)

		return f_local * (1 + gamma) + beta

@MODEL.register_module
def fusion(pretrained=False, **kwargs):
	model = GlobalFiLM(**kwargs)
	return model

''''========== ViT Encoder =========='''
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
			print('Try load model for LMAD')
			time.sleep(1)
	return model

@MODEL.register_module
def mask_vit_small_patch16_224_dino(pretrained=True, **kwargs):
	""" ViT-Small-DINO, patch16"""
	model_kwargs = dict(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=384, depth=12, num_heads=6)
	model = create_vision_transformer('vit_small_patch16_224.dino', pretrained=pretrained, **dict(model_kwargs, **kwargs))
	return model

@MODEL.register_module
def mask_vit_small_patch8_224_dino(pretrained=True, **kwargs):
	""" ViT-Small-DINO, patch8"""
	model_kwargs = dict(img_size=kwargs.pop('img_size'), patch_size=8, embed_dim=384, depth=12, num_heads=6)
	model = create_vision_transformer('vit_small_patch8_224.dino', pretrained=pretrained, **dict(model_kwargs, **kwargs))
	return model


'''========================================== ViT Decoder ================================================'''
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
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # [3, B, num_heads, N, head_dim]
		q, k, v = qkv.unbind(0) # [B, num_heads, N, head_dim]
		q, k = self.q_norm(q), self.k_norm(k)

		q = q * self.scale
		attn = q @ k.transpose(-2, -1) # [B, num_heads, N, N]
		# print('Attention中attn的维度是：', attn.shape)

		if attn_mask is not None:
			attn = attn + attn_mask.unsqueeze(1)

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
		# print('TransformerBlock的输入x: ', x.shape)
		x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
		x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
		return x

class DecoderLayer(nn.Module):
	def __init__(self, dim, num_heads):
		super(DecoderLayer, self).__init__()
		self.block1 = TransformerBlock(dim=dim, num_heads=num_heads)
		self.block2 = TransformerBlock(dim=dim, num_heads=num_heads)
		self.block3 = TransformerBlock(dim=dim, num_heads=num_heads)

	def forward(self, x, mask=None):
		# x: [B, L, C]
		out = self.block1(x, mask)
		out = self.block2(out, mask)
		out = self.block3(out, mask)

		return out

class Decoder(nn.Module):
	def __init__(self, dim, num_heads, img_size=256, patch_size=8, in_chans=3, embed_dim=384, drop_rate=0., pre_norm=False):
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

		self.layer_1 = DecoderLayer(dim=dim, num_heads=num_heads)
		self.layer_2 = DecoderLayer(dim=dim, num_heads=num_heads)
		self.layer_3 = DecoderLayer(dim=dim, num_heads=num_heads)

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

	def forward(self, x, mask=None):
		# x: [B, L, C]
		B, L, C = x.shape
		H = int(np.sqrt(L))

		if self.training:
			x = self.add_jitter(feature_tokens=x, scale=20.0, prob=1.0)

		# x = self.patch_embed(x)
		x = x + self.pos_embed
		x = self.pos_drop(x)

		out = []
		layer_out = self.layer_1(x, mask=None)
		_layer_out = layer_out.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
		out.append(_layer_out)

		layer_out = self.layer_2(layer_out, mask=None)
		_layer_out = layer_out.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
		out.append(_layer_out)

		layer_out = self.layer_3(layer_out, mask=None)
		_layer_out = layer_out.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
		out.append(_layer_out)

		return [out[int(len(out)-1-i)] for i in range(len(out))]
		# return out
@MODEL.register_module
def create_decoder(pretrained=False, **kwargs):
	model = Decoder(**kwargs)
	return model

# ========== ViTAD ==========
class ViTAD(nn.Module):
	def __init__(self, model_t1, model_t2, model_s):
		super(ViTAD, self).__init__()
		self.net_t1 = get_model(model_t1)	# [B, 384, 32, 32]
		self.net_t2 = get_model(model_t2)	# [B, 384, 16, 16]

		self.film = GlobalFiLM(dim=384)

		self.net_s = get_model(model_s)


		self.frozen_layers = ['net_t1', 'net_t2']

	def generate_mask(self, imgs):

		mask = self.Lmask(imgs)

		B, H, W = mask.shape
		N = H * W
		mask_flat = mask.view(B, H*W)

		attn_mask = mask_flat[:, None, :] * mask_flat[:, :, None]  # [B, 256, 256]
		attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
		for b in range(B):
			row_sum = attn_mask[b].sum(dim=1)
			for i in range(row_sum.size(0)):
				if row_sum[i] == -float('inf') * N:
					attn_mask[b, i, i] = 0.0  # 保留自己关注自己

		attn_mask = attn_mask.masked_fill(attn_mask != -float('inf'), 0.)
		return attn_mask

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
		B = imgs.shape[0]
		# encoder
		feats_fine, feats_fine_n = self.net_t1(imgs)	# [B, 384, 32, 32]
		feats_fine = [f.detach() for f in feats_fine]
		feats_fine_n = [f.detach() for f in feats_fine_n]

		feats_coarse, _ = self.net_t2(imgs)		# [B, 384, 16, 16]
		feats_coarse = [f.detach() for f in feats_coarse]
		# print('teacher_cls: ', teacher_cls.shape)	# [B, 384]

		feat_local = feats_fine_n[0]
		feat_global = feats_coarse[0]

		# proto = feat_global.mean(dim=[2, 3], keepdim=True)
		# semantic_map = ((feat_global - proto) ** 2).sum(dim=1)  # [B, 16, 16]
		# semantic_map = (semantic_map - semantic_map.min()) / (semantic_map.max() - semantic_map.min() + 1e-6)

		de_input = self.film(f_local=feat_local, f_global=feat_global) # [B, 384, 32, 32]

		# decoder
		de_feats = self.net_s(de_input)

		return feats_fine, de_feats

@MODEL.register_module
def glmad(pretrained=False, **kwargs):
	model = ViTAD(**kwargs)
	return model


if __name__ == '__main__':
	from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
	from util.util import get_timepc, get_net_params
	from argparse import Namespace as _Namespace

	bs = 8
	reso = 256
	x = torch.randn(bs, 3, reso, reso).cuda()

	model_t1 = _Namespace()
	model_t1.name = 'mask_vit_small_patch8_224_dino'
	model_t1.kwargs = dict(pretrained=True, checkpoint_path='', strict=True, img_size=reso, teachers=[3, 6, 9], neck=[12])
	model_t2 = _Namespace()
	model_t2.name = 'mask_vit_small_patch16_224_dino'
	model_t2.kwargs = dict(pretrained=True, checkpoint_path='', strict=True, img_size=reso, teachers=[12], neck=[])

	model_s = _Namespace()
	model_s.name = 'create_decoder'
	model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=False, dim=384, num_heads=6, img_size=reso)
	model = _Namespace()
	model.name = 'glmad'
	model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_t1=model_t1, model_t2=model_t2, model_s=model_s)

	net = glmad(model_t1=model_t1, model_t2=model_t2, model_s=model_s).cuda()
	net.eval()
	# net.train()
	feats_fine, de_feats = net(x)
	print('feats_fine len: ', len(feats_fine))
	print('feats_fine: ', feats_fine[0].shape, feats_fine[1].shape, feats_fine[2].shape)
	print('de_feats: ', de_feats[0].shape, de_feats[1].shape, de_feats[2].shape)
	# print('semantic_map: ', semantic_map.shape)

