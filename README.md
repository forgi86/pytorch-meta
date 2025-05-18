# Meta learning in Pytorch

This repository provides minimial documented implementations of basic meta learning algorithms in PyTorch.

They are based on the recent [functional API](https://docs.pytorch.org/docs/stable/func.api.html) of PyTorch, using transforms like `vmap` and `grad`. 

## Contents

- `maml_pytorch.ipynb`: Model-Agnostic Meta Learning algorithm

- `hypernet_pytorch.ipynb` Black-box Meta Learning using a hypernet to learn the parameters of an MLP


Both examples solve the toy regression problem with sinusoidal signals of different phase and amplitudes introduced in the [MAML paper](https://arxiv.org/pdf/1703.03400).

## Related Repositories

I provide very similar JAX-based implementations of the same meta learning algorithms in [this repo](https://github.com/forgi86/jax-tutorial). 

