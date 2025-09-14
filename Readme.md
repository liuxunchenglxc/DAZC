# Code for Review

## Build $X^n$ by clustering

All in the script `labelcluster/labelcluster.py`

## Search Process

All in the script `search.sh`.

## Retrain Process

Use the supernet_train.py, the same as AutoFormer.

- Hyperparameter of retaining from AutoFormer checkpoint: `--epochs 300 --warmup-epochs 20 --lr 2e-6 --min-lr 1e-8 --batch-size 256`

## Transfer Learning

Use the supernet_train.py.

- Hyperparameter for C10: `--epochs 300 --warmup-epochs 10 --lr 2.2e-4 --min-lr 2.2e-6 --batch-size 64`
- Hyperparameter for C100: `--epochs 300 --warmup-epochs 10 --lr 7e-5 --min-lr 7e-7 --batch-size 64`
- Hyperparameter for FLOWERS: `--epochs 3000 --warmup-epochs 100 --lr 7e-5 --min-lr 7e-7 --batch-size 256`
- Hyperparameter for CARS: `--epochs 1500 --warmup-epochs 50 --lr 7e-3 --min-lr 7e-5 --batch-size 64`

## What is our implementation based on?

The code from https://github.com/microsoft/Cream/tree/main/AutoFormer. The MIT license of AutoFormer is put in `license/AutoFormer`.

### `evolution.py`

We implement our DAZC process in this file, and we change the original process to the training-free behavior.

### `supernet_engine.py` and `proxies/*`

We implement the proxies in `proxies/*` and add zero-cost proxy support for the `supernet_engine.py` .

### `supernet_train.py`

We add few lines of codes for transfer learning.

### `labelcluster/*`

Implement for building $X^n$.

### Other files from AutoFormer

Some bug-fix when running with the new PyTorch version.

## How to build the experimental environment?

Check the `Dockerfile` or `labelcluster/Dockerfile`, then, `docker build -t image_name .` or `docker build -t image_name labelcluster`. Finally, mount this code project into container by `docker run -v this_path:inner_path ...`.