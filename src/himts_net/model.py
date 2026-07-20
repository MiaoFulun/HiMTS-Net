import torch
from torch import nn


class ResidualContextBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.depthwise(values)
        values = self.pointwise(values)
        values = self.norm(values)
        values = self.activation(values)
        values = self.dropout(values)
        return values + residual


class HierarchicalEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 32,
        kernels: tuple[int, int, int] = (15, 21, 31),
        blocks_per_stage: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        widths = (hidden_dim, hidden_dim * 2, hidden_dim * 2)
        self.stem = nn.Conv1d(in_channels, widths[0], kernel_size=1)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        ResidualContextBlock(width, kernel_size=kernel, dropout=dropout)
                        for _ in range(blocks_per_stage)
                    ]
                )
                for width, kernel in zip(widths, kernels)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(widths[index], widths[index + 1], kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                )
                for index in range(len(widths) - 1)
            ]
        )
        self.projection = nn.Conv1d(widths[-1], hidden_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.stem(values)
        for index, stage in enumerate(self.stages):
            values = stage(values)
            if index < len(self.downsamples):
                values = self.downsamples[index](values)
        values = self.projection(values)
        return self.pool(values).squeeze(-1)


class HiMTSNet(nn.Module):
    def __init__(self, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.branch_names = ("time", "local_dse", "entropy_contribution", "sms")
        self.branches = nn.ModuleDict(
            {
                "time": HierarchicalEncoder(3, hidden_dim=hidden_dim, dropout=dropout),
                "local_dse": HierarchicalEncoder(1, hidden_dim=hidden_dim, dropout=dropout),
                "entropy_contribution": HierarchicalEncoder(
                    1,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                ),
                "sms": HierarchicalEncoder(1, hidden_dim=hidden_dim, dropout=dropout),
            }
        )
        merged_dim = hidden_dim * len(self.branch_names)
        self.gate = nn.Sequential(
            nn.Linear(merged_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.branch_names)),
            nn.Softmax(dim=1),
        )
        self.head = nn.Sequential(
            nn.Linear(merged_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        time: torch.Tensor,
        local_dse: torch.Tensor,
        entropy_contribution: torch.Tensor,
        sms: torch.Tensor,
    ) -> torch.Tensor:
        inputs = {
            "time": time,
            "local_dse": local_dse,
            "entropy_contribution": entropy_contribution,
            "sms": sms,
        }
        embeddings = [self.branches[name](inputs[name]) for name in self.branch_names]
        merged = torch.cat(embeddings, dim=1)
        weights = self.gate(merged)
        fused = torch.cat(
            [
                embedding * weights[:, index : index + 1]
                for index, embedding in enumerate(embeddings)
            ],
            dim=1,
        )
        return self.head(fused).squeeze(-1)

