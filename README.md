# HiMTS-Net

PyTorch implementation of the network architecture in **HiMTS-Net: A Native-Axis-Preserving Hierarchical Multi-Domain Time-Spectral Network for Small Target Detection in Sea Clutter**.

**Authors:** Fulun Miao, Huayu Fan, Yi Liu, Yuanshuai Li, Guanqun Wang, Baogui Qi, and Quanhua Liu.

HiMTS-Net uses four independent hierarchical 1D encoders to process the complex slow-time echo, local Doppler spectral entropy (DSE), full-window entropy contributions, and STFT marginal spectrum (SMS) along their native axes. A sample-adaptive gate combines the branch embeddings for binary target detection.

![HiMTS-Net architecture](assets/method_overview.png)

## Quick start

Use Python 3.10 or newer. Install the dependency from the repository directory:

```bash
python -m pip install -r requirements.txt
```

The model accepts four prepared tensors in `[batch, channels, length]` format. For a 512-pulse window with 64-pulse frames and a stride of 16, an example forward pass is:

```python
import torch
from model import HiMTSNet

model = HiMTSNet(hidden_dim=32, dropout=0.1).eval()
batch_size = 2

with torch.no_grad():
    logits = model(
        time=torch.randn(batch_size, 3, 512),  # real, imaginary, magnitude
        local_dse=torch.randn(batch_size, 1, 29),  # sliding frames
        entropy_contribution=torch.randn(batch_size, 1, 512),  # Doppler bins
        sms=torch.randn(batch_size, 1, 64),  # Doppler bins
    )

print(logits.shape)  # torch.Size([2])
```

The output contains one raw logit per sample; `torch.sigmoid(logits)` converts logits to scores between 0 and 1.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grants 62371046 and 62388102.

## License

[MIT](LICENSE).
