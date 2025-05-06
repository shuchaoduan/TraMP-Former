import os
from collections import OrderedDict

import torch
from torch import nn

from models.RGB.ST_Former import FormerDFER
from models.fusion.vit_decoder_1 import decoder_fuser_1
from models.reg_heads.regress_head import MLP_head
from models.encoders.SkateFormer_DFEW_rgb1x1 import SkateFormer_DFEW_rgb1x1



class Quality_Model(nn.Module):
    def __init__(self, args, channels, **kwargs):
        super().__init__()
        self.args = args
        self.mask = args.mask_pad
        self.fuser = args.fuser
        self.clip_len = args.clip_len

        self.encoder = SkateFormer_DFEW_rgb1x1(channels=channels, embed_dim= channels[0], **kwargs)
        self.rgb_encoder = FormerDFER()

        self.fusion = decoder_fuser_1(dim=channels[-1], num_heads=8, num_layers=3, drop_rate=0.)
        self.evaluator = MLP_head(channels[-1])
        self.rgb_transform = nn.Linear(channels[-1] * 2, channels[-1])

    def load_pretrained(self, ckpt_path):

        rgb_ckpt_path = os.path.join(ckpt_path, 'FormerDFER_DFEW.pth')
        rgb_model_state_dict = OrderedDict()
        rgb_ckpt = torch.load(rgb_ckpt_path, map_location='cpu')
        for k, v in rgb_ckpt['state_dict'].items():
            name = k[7:]
            if 't_former.spatial_transformer' in name:
                new_k = name.replace('t_former.spatial_transformer', 't_former.temporal_transformer')
                rgb_model_state_dict[new_k] = v
            else:
                rgb_model_state_dict[name] = v
        rgb_model_state_dict.pop('fc.weight')
        rgb_model_state_dict.pop('fc.bias')
        self.rgb_encoder.load_state_dict(rgb_model_state_dict, strict=False)

        traj_ckpt_path = os.path.join(ckpt_path, 'SkateFormer_256_rgb1x1.pth')
        traj_model_state_dict = OrderedDict()
        traj_ckpt = torch.load(traj_ckpt_path, map_location='cpu')

        for k, v in traj_ckpt['model'].items():
            name = k[7:]
            traj_model_state_dict[name] = v
        traj_model_state_dict.pop('evaluator.head.weight')
        traj_model_state_dict.pop('evaluator.head.bias')
        self.encoder.load_state_dict(traj_model_state_dict, strict=False)
        print('DFEW-pretained weights loaded')

    def forward(self, rgb, trajs, index_t, trajs_len=None):

        N, C, T, H, W = rgb.shape
        rgb_feats = self.rgb_encoder(rgb)
        rgb_feats = rgb_feats.reshape(N, -1, 512) # [b,5, 512]

        traj_feats = self.encoder(trajs, index_t) # [b, 256]

        traj_feats = traj_feats.unsqueeze(1)
        rgb_feats= self.rgb_transform(rgb_feats).mean(1).unsqueeze(1)

        fusion_feats = self.fusion(rgb_feats, traj_feats).mean(1)
        prediction = self.evaluator(fusion_feats)

        return prediction, traj_feats, rgb_feats