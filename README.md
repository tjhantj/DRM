# DRM

This repository proposes the source code in the following paper submitted to CIKM 2026

## Build

```bash
cd code
make
```

This produces a single binary `drm`. Requires a C++17 compiler.

## Run

From the `code` directory:

```bash
./drm --network karate --output run1 --algorithm MGA \
      --phi cosine --gamma 1.0 --sampling 0.3 --g 1
```

Required:

| Flag | Description |
|---|---|
| `--network`    | Dataset name (folder under `datasets/`) |
| `--output`     | Output sub-directory under `output/` |

Optional:

| Flag | Default | Choices |
|---|---|---|
| `--algorithm` | `MGA` | `MGA`, `BGD` |
| `--phi`       | `cosine` | `cosine`, `pearson`, `sigmoid` |
| `--gamma`     | `1.0` | resolution parameter |
| `--sampling`  | `0.3` | fraction of node pairs sampled for `g(d)` |
| `--g`         | `1`   | main method or ablation — see below |

`--g` selects how `g(d)` is realized:

| Value | Name | Meaning |
|---|---|---|
| `1`  | `kmeans`   | main method:  k-means on sampled distances |
| `2`  | `nog`      | ablation: `g(d) ≡ 1` |
| `3`  | `nokk2m`   | ablation: drop the `k_u k_v / 2m` factor |
| `4`  | `nophi`    | ablation: drop `phi` from the observed term (`w_uv = A_uv`) |
| `5`  | `gaverage` | ablation: `g(d) ≡` mean phi over edges |

## Datasets

Each dataset directory under `datasets/` contains:

```
network.dat     edge list:           u v [w]
community.dat   ground-truth labels: node label
embedding.dat   node embeddings:     node f1 f2 ... fd
feature.npy     raw node features (attributed datasets only)
```

`embedding.dat` is what the binary reads. `feature.npy` is provided for
the attributed datasets (`cora`, `citeseer`, `amazon_pc`) as the raw
input from which the embeddings were derived; it is not consumed by the
binary directly but is included for reproducibility.

Provided datasets: `karate`, `cora`, `citeseer`, `amazon_pc`,
`amazon`, `dblp`, `youtube`.

**Note.** Due to GitHub's file size limit, `embedding.dat` for `youtube` is not included in this repository.
You can generate them using unsupervised graph embedding methods from [AutoGL](https://github.com/THUMNLab/AutoGL/).

## Output

Each run writes:

```
output/<output_dir>/<run_name>/<idx>.txt    # node-to-community assignment
output/<output_dir>/summary.csv              # parameters + timing
```

`<run_name>` encodes the parameters (algorithm, phi, gamma, etc.) so
that repeated runs of the same configuration are grouped in one folder
and indexed sequentially.
