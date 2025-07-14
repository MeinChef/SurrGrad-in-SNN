# SurrGrad-in-SNN
Analysing learning rules of SNNs

## NMNIST
For performance reference:

No optimisation, single threaded: 90s for dataprep, 1 epoch, 50 iterations; GPU between 0 and 20%

5 worker, prefetch: 36s, GPU at ~30%

5 worker, 20 prefetch: 35s, GPU at ~40%, measurement error

### How to use
- Make sure the working directory of the python venv is in fact the top folder of the git repository.

### Next TODO
- During testing, I have encountered only class 0 and 1. Investigate data and data preparation thuroughly. - Solved, faulty cache directory.
- When training on temporal data, the loss is around 0.9 from the get-go
    - For mse, the loss is 0.9935 for _every single datapoint_
    - For ce the loss is about 2.3036 for _every single datapoint_
- During training on the rate data, loss decreases to ~8 until like batch 27, and then jumps to ~38 and stays exactly there
- Visualize what the network is doing