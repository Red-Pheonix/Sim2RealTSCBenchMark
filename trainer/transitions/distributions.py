class Distribution:
    def sample(self, rng):
        raise NotImplementedError


class UniformDistribution(Distribution):
    def __init__(self, config):
        self.low = config["low"]
        self.high = config["high"]

    def sample(self, rng):
        return rng.uniform(self.low, self.high)


class NormalDistribution(Distribution):
    def __init__(self, config):
        self.mean = config.get("mean", 0.0)
        self.std = config.get("std", 1.0)

    def sample(self, rng):
        return rng.normal(self.mean, self.std)
