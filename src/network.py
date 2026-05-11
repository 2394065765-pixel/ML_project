import torch.nn as nn


class BuildingLoadModel(nn.Module):
    """
    建筑恒载预测模型（多输出回归）
    输出：
        1. 整层构建总自重
        2. 整体合计恒载
    """

    def __init__(self, input_dim, output_dim=2, dropout=0.1):
        super(BuildingLoadModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)