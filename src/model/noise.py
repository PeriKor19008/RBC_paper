
from src.model.RBCDataset import *


class WithTransform(Dataset):
    def __init__(self, base, transform=None):
        self.base = base
        self.transform = transform
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return (self.transform(x), y) if self.transform else (x, y)


class AddGaussianNoise(nn.Module):
    def __init__(self, pct: float = 0.05, p: float = 0.5):
        """
        pct: The percentage of noise to add (0.05 = ~5% variation)
        p: Probability of applying the noise (50%)
        """
        super().__init__()
        self.pct = float(pct)
        self.p = float(p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Applies noise 50% of the time
        if torch.rand(()) < self.p:
            # Multiplies the image by numbers like 0.96, 1.04, 0.99...
            noise_multiplier = 1.0 + (torch.randn_like(x) * self.pct)
            return x * noise_multiplier

        return x


class AddSpeckleNoise(nn.Module):
    def __init__(self, std: float = 0.05, p: float = 0.5):
        super().__init__()
        self.std = float(std)
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < self.p:
            return x + x * (torch.randn_like(x) * self.std)
        return x






