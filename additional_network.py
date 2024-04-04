import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k=6):
        super(TNet, self).__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, k*k)
        )
        self.fc.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mlp(x)
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = self.fc(x)
        x = x.view(batch_size, self.k, self.k)
        return x

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.input_transform = TNet(k=6)
        self.feature_extraction = nn.Sequential(
            nn.Conv1d(6, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 512, 1),  # 直接输出每个点的512维特征
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.transpose(2, 1)
        # Input transform
        trans = self.input_transform(x)
        x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)
        
        # Feature extraction
        x = self.feature_extraction(x)
        
        # 在这一步不使用全局池化，保持每个点的特征
        x = x.transpose(2, 1)  # 转换回[batch_size, num_points, features]的格式
        
        return x
    
    
class ComplexMLP(nn.Module):
    def __init__(self):
        super(ComplexMLP, self).__init__()
        # 第一层隐藏层
        self.fc1 = nn.Linear(512, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.2)        
        # 第二层隐藏层
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.2)        
        # 第三层隐藏层
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.2)        
        # 第四层隐藏层
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.2)        
        # 输出层
        self.fc5 = nn.Linear(128, 14)

    def forward(self, x):
        x = self.dropout1(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.leaky_relu(self.bn4(self.fc4(x))))
        output = self.fc5(x)

        # 按指定维度拆分输出
        # means3D = output[:, :3]
        # means2D = output[:, 3:6]
        # shs = output[:, 6:9]
        # opacity = output[:, 9:10]
        # scales = output[:, 10:13]
        # rotations = output[:, 13:17]

        return output