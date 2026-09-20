import torch


class ReturnMetrics:
    """Streaming, sample-weighted metrics in unscaled log-return units."""
    def __init__(self, device):
        self.sums = torch.zeros(4, 3, dtype=torch.float64, device=device)
        self.count = 0

    def update(self, prediction, target):
        p, y = prediction.detach().double(), target.detach().double()
        self.sums[0] += ((p-y)**2).sum(0)
        self.sums[1] += (p-y).abs().sum(0)
        self.sums[2] += y.sum(0)
        self.sums[3] += (y*y).sum(0)
        self.count += len(y)

    def compute(self):
        if not self.count:
            raise ValueError("Cannot evaluate an empty dataset")
        s = self.sums.cpu()
        mse = s[0]/self.count
        sst = s[3]-s[2]**2/self.count
        return dict(samples=self.count, mse=mse.tolist(), rmse=mse.sqrt().tolist(),
                    mae=(s[1]/self.count).tolist(),
                    r2=[float(1-s[0, i]/sst[i]) if sst[i] > 0 else None for i in range(3)],
                    mean_mse=float(mse.mean()))
