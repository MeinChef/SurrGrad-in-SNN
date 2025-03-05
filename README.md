# SurrGrad-in-SNN
Analysing learning rules of SNNs

## NMNIST
For performance reference:

No optimisation, single threaded: 90s for dataprep, 1 epoch, 50 iterations; GPU between 0 and 20%

5 worker, prefetch: 36s, GPU at ~30%

5 worker, 20 prefetch: 35s, GPU at ~40%, measurement error

### How to use
- Make sure the working directory of the python venv is in fact the top folder of the git repository.