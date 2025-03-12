import torch
import numpy as np
# a = torch.rand([1000, 3, 3])
# print(a[..., None, :])
# print(a[..., np.newaxis, :].shape)


max_freq = 20 - 1
N_freqs = 20

freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)


embed_fns = []
periodic_fn = [torch.sin, torch.cos]

for freq in freq_bands:
    for p_fn in periodic_fn:
        embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))

        
inputs = torch.ones(20, 3 + 3)


res = torch.cat([fn(inputs) for fn in embed_fns], -1)
print(res)
print(res.shape)

6 * 20 ( 2)

print(torch.sin(torch.tensor([1, 2, 3])))

