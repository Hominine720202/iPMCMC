def init_proposal(sampler, params):
    return lambda n: sampler(**params, size=n)