import torch
import torch.nn as nn


class NTU_Fi_CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super(NTU_Fi_CNN_LSTM, self).__init__()

        # CNN模块，用于特征提取
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=12, stride=6),  # 1 input channel, 16 output channels
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 2x MaxPool
            nn.Conv1d(16, 32, kernel_size=7, stride=3),  # 16 input channels, 32 output channels
            nn.ReLU(),
        )

        # 计算CNN输出的维度后进行LSTM的输入，8是CNN特征的尺寸
        self.lstm_input_size = 32  # CNN输出通道数
        self.lstm_hidden_size = 128
        self.num_layers = 1

        # LSTM模块，处理时序特征
        self.lstm = nn.LSTM(32, 128, 1, batch_first=True)

        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(self.lstm_hidden_size, num_classes),  # LSTM输出的hidden_size作为全连接层输入
            nn.Softmax(dim=1)  # 输出类别概率
        )

    def forward(self, x):
        batch_size = x.size(0)

        # CNN部分 - 特征提取
        x = x.view(batch_size, 1, -1)  # 调整输入形状 (batch_size, 1, feature_size)
        x = self.encoder(x)  # 输出大小：(batch_size, 32, new_length)

        # LSTM部分 - 处理时序数据
        x = x.permute(0, 2, 1)  # (batch_size, new_length, 32)
        lstm_out, (ht, ct) = self.lstm(x)  # ht: 最后一个时刻的hidden state

        # 使用LSTM最后一层的hidden state进行分类
        outputs = self.classifier(ht[-1])  # ht[-1]是最后一层LSTM的输出

        return outputs