class Distribution:
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def rvs(self, given=None, **kwargs):
        raise NotImplementedError

    def logpdf(self, given=None, **kwargs):
        raise NotImplementedError

    def pdf(self, *args, given=None, **kwargs):
        return np.exp(self.logpdf(*args, given=given, **kwargs))

    def sample(self, *args, **kwargs):
        return self.rvs(*args, **kwargs)

    def density(self, *args, **kwargs):
        return self.pdf(*args, **kwargs)
