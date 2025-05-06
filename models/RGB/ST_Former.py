import torch
from torch import nn
from models.RGB.S_Former import spatial_transformer
from models.RGB.T_Former import temporal_transformer


class FormerDFER(nn.Module):
    def __init__(self,cls_num=7):
        super().__init__()
        self.s_former = spatial_transformer()
        self.t_former = temporal_transformer()
        # self.fc = nn.Linear(512, cls_num)

    def forward(self, x):
        n_batch, frames, _, _, _ = x.shape
        n_clips = int(frames/16)
        # split video sequence into n segments and pack them
        if frames>16:
            data_pack = torch.cat([x[:,i:i+16] for i in range(0, frames-1, 16)])
            # out_s = self.s_former(data_pack)
            out_s= self.s_former(data_pack)
        else:
            # out_s = self.s_former(x)
            out_s= self.s_former(x)# []
        out_t = self.t_former(out_s)
        # if PD
        # out_t = out_t.reshape(n_batch,n_clips,-1)
        # out_avg = out_t.mean(dim=1)
        # out_fc = self.fc(out_avg)
        return out_t

# def freeze_s_former(model):
#     model.s_former.conv1.requires_grad = False
#     model.s_former.bn1.requires_grad = False
#     model.s_former.relu.requires_grad = False
#     model.s_former.maxpool.requires_grad = False
#     model.s_former.layer1.requires_grad = False
#     model.s_former.layer2.requires_grad = False
#     model.s_former.layer3.requires_grad = False
#
#     return model
    # for name, p in model.named_parameters():
    #     if 's_former' in name:
    #         p.requires_grad = False


if __name__ == '__main__':
    img = torch.randn((2, 80, 3, 224, 224))
    model = FormerDFER(cls_num=5)
    model(img)
