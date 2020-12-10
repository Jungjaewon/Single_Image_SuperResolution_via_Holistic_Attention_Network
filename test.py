import torch
import torch.nn as nn

class Channel_Spatial_Attention_Module(nn.Module):
    def __init__(self):
        super(Channel_Spatial_Attention_Module, self).__init__()
        self.conv_3d = nn.Conv3d(1, 1, kernel_size=3, padding=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n, c, h, w = x.size()
        x_reshape = x.reshape(n, 1, c, h, w)
        x_3d = self.sigmoid(self.conv_3d(x_reshape))
        x_squzzed = x_3d.reshape(n, c, h, w)
        return (self.scale * x_squzzed) * x + x


class Layer_Attention_Module(nn.Module):
    def __init__(self):
        super(Layer_Attention_Module, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, feature_group):
        b,n,c,h,w = feature_group.size()
        feature_group_reshape = feature_group.view(b, n, c * h * w)

        attention_map = torch.bmm(feature_group_reshape, feature_group_reshape.view(b, c * h * w, n))
        attention_map = self.softmax(attention_map) # N * N

        attention_feature = torch.bmm(attention_map, feature_group_reshape) # N * CHW
        b, n, chw = attention_feature.size()
        attention_feature = attention_feature.view(b,n,c,h,w)

        attention_feature = self.scale * attention_feature + feature_group
        b, n, c, h, w = attention_feature.size()
        return attention_feature.view(b, n * c, h, w)


if __name__ == '__main__':
    pass

    base = list()
    for _ in range(10):
        base.append(torch.rand((3,5,10,10)))


    #print(torch.stack(base, dim=1).size())

    CSA = Channel_Spatial_Attention_Module()
    LA = Layer_Attention_Module()

    print(CSA(base[-1]).size())
    print(LA(torch.stack(base, dim=1)).size())

    feature_csa = CSA(base[-1])
    feature_la = LA(torch.stack(base, dim=1))

    feature_csa + feature_la