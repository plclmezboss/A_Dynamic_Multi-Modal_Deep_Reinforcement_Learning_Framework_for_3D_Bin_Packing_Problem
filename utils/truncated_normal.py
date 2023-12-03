#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch


def parameterized_truncated_normal(uniform, mu, sigma, a, b):
    normal = torch.distributions.normal.Normal(0, 1)

    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    alpha_normal_cdf = normal.cdf(alpha)
    p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * uniform

    p = p.numpy()
    one = np.array(1, dtype=p.dtype)
    epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
    v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
    x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))
    x = torch.clamp(x, a, b)

    return x


def truncated_normal(uniform, mu, sigma, a, b):
    return parameterized_truncated_normal(uniform, mu, sigma, a, b)


def sample_truncated_normal(shape=(), mu=0.4, sigma=1.0,  a=0.02, b=2.0):
    return truncated_normal(torch.from_numpy(np.random.uniform(0, 1, shape)).float(), mu, sigma, a, b)

def generate_normal(shape=(), mu=0.4, sigma=1.0, a=0.02, b=2.0):
    dataset = []
    for i in range(shape[0] * shape[1]):
        data = torch.FloatTensor(1).normal_(mean=mu, std=sigma)
        # while not a <= data <= b:
		#     data = torch.FloatTensor(1).normal_(mean=mu, std=sigma)
        # dataset.append(data)
        while not a <= data <= b:
            data = torch.FloatTensor(1).normal_(mean=mu, std=sigma)
        dataset.append(data)
    dataset = torch.stack(dataset, 0)
    dataset = torch.reshape(dataset, shape)
    return dataset
    