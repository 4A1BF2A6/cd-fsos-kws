class AverageValueMeter:
    """Minimal torchnet.meter.AverageValueMeter replacement."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.mean = 0.0
        self.std = float("nan")

    def add(self, value, n=1):
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            value = value.item()
        value = float(value)
        n = int(n)
        if n <= 0:
            return

        old_n = self.n
        old_mean = self.mean
        self.n += n
        self.sum += value * n
        self.mean = self.sum / self.n
        if old_n == 0:
            self.var = 0.0
        else:
            self.var += n * (value - old_mean) * (value - self.mean)
        self.std = (self.var / max(self.n - 1, 1)) ** 0.5

    def value(self):
        return self.mean, self.std
