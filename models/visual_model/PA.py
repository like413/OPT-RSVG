import  torch
import torch.nn as nn
from torch.nn import functional as F
class PA(nn.Module):
    def __init__(self, in_channels, out_channels, feature_dim,k,s):
        super(PA, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.gamma_mlp = nn.Sequential(nn.Linear(feature_dim, in_channels), nn.Tanh(), nn.Linear(in_channels, out_channels))
        self.beta_mlp = nn.Sequential(nn.Linear(feature_dim, in_channels), nn.Tanh(), nn.Linear(in_channels, out_channels))
        self.rafa_mlp = nn.Sequential(nn.ConvTranspose2d(1, 1, kernel_size=5, stride=1, padding=0), nn.Tanh(),  nn.ConvTranspose2d(1, 1, kernel_size=k, stride=s))
        #self.conv = nn.Conv2d(in_channels,out_channels,1)

    def forward(self, x, language_features):
        batch_size, channels, height, width = x.size()
        language_features = self.avgpool(language_features.permute(0, 2, 1)).squeeze(-1)  # 计算全局语言特征
        ef = language_features
        gamma = self.gamma_mlp(ef).view(batch_size, -1, 1, 1)  # 计算缩放参数
        #x = self.conv(x)
        beta = self.beta_mlp(ef).view(batch_size, -1, 1, 1)  # 计算平移参数
        #rafa = self.rafa_mlp(ef).view(batch_size, -1, 1, 1)  #
        out1 = gamma * x + beta  # 应用PA操作


        weight = ef.view(batch_size,16,16).unsqueeze(1)
        weight = self.rafa_mlp(weight)

        out2 = x*weight
        out = F.relu(out1+out2)
        return out
if __name__ == '__main__':
    model = PA(256,256,768)
    x = torch.randn(1, 256, 160, 160)
    t = torch.randn(1,20,768)
    y = model(x,t)
    print(y.size())