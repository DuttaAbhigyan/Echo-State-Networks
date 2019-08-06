# Echo-State-Networks
Implementation of Echo State Networks. In this repository I try to recreate the Echo State Network form the code
form https://mantas.info/code/simple_esn/ by the author Dr. Mantas Lukoševičius and modify it to make it more modular
and easy to generalize. The ideas are taken form the paper by the same author:

Mantas Lukoševičius:
A practical guide to applying Echo State Networks

The data to test our model working is the same as used by the aforementioned author i.e Mackey Glass data

## Modifications:

- Made the ESN more modular and provided easy to tune hyperparameters
- Implementations for both already initialized starting state (for testing) and also
  non-initialized starting state (0's) are provided.
