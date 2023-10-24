import torch.nn as nn
import torch.nn.functional as F
import torch

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock5x5(nn.Module):  # for CNN6
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(5, 5), stride=stride,
                               padding=(2, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    # 初始化卷积层的权重和批量归一化层的参数。
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x

class CNN6(nn.Module):
    def __init__(self, num_classes=4, do_dropout=False, embed_only=False):
        super(CNN6, self).__init__()

        self.embed_only = embed_only  # 标识是否只返回嵌入特征而不进行最终的分类
        self.do_dropout = do_dropout  # 标识是否在卷积层后应用了 dropout。
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64, stride=(1,1))
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128, stride=(1,1))
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256, stride=(1,1))
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512, stride=(1,1))
        # 如果 self.do_dropout 为真，则会应用一个 dropout 层，以防止过拟合
        self.dropout = nn.Dropout(0.2)
        # 全连接层，用于最终的分类任务。它将卷积块的输出映射到 num_classes 个类别。
        self.linear = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)

        x = torch.mean(x, dim=3)  # mean over time dim
        (x1, _) = torch.max(x, dim=2)  # max over freq dim
        x2 = torch.mean(x, dim=2)  # mean over freq dim (after mean over time)
        x = x1 + x2

        if self.embed_only:
            return x
        return self.linear(x)

# P_h = (K_h - 1) / 2
# P_w = (K_w - 1) / 2
class AudioCNN(nn.Module):
    def __init__(self, num_classes=4):  # 根据您的任务设置类的数量
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=9, stride=1, padding=4)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.bn3 = nn.BatchNorm1d(32)

        # Global Max Pooling layer
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        print(x.shape)
        # x = x.permute(1, 2, 0)
        # print(x.shape)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 8)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 8)

        x = F.relu(self.bn3(self.conv3(x)))

        # Global Max Pooling
        x = self.global_max_pool(x)
        x = torch.squeeze(x, -1)

        x = self.fc(x)
        # print("x shape: ", x.shape)
        return x


class LinearClassifier(nn.Module):
    def __init__(self, num_classes=4, device="cpu"):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(512, num_classes).to(device)

    def forward(self, features):
        return self.fc(features)



