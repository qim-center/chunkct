# chunkct

Chunked cone-beam CT reconstruction (FDK) built on the
[Core Imaging Library (CIL)](https://github.com/TomographicImaging/CIL).
Projections are chunked along the angle dimension and the reconstruction volume
is chunked into blocks, so scans far larger than GPU or host memory can be
reconstructed, either on a single GPU machine or as LSF batch jobs on the
[DTU HPC cluster](https://www.hpc.dtu.dk/).
Projections and the reconstruction are stored as [Zarr](https://zarr.dev) arrays.

## Installation

chunkct needs CIL and its GPU backend TIGRE, which are only available as conda packages
(not on PyPI). pixi installs conda packages, so with pixi they are included; with pip you
first install them with conda.

**With pixi (recommended)**

```sh
git clone https://github.com/qim-center/chunkct.git
cd chunkct
pixi install          # CIL + TIGRE (CUDA 12), plus test, lint and Jupyter tools
```

On a DTU HPC login node (no GPU), pixi refuses to install the CUDA environment unless
told to assume CUDA 12: `CONDA_OVERRIDE_CUDA=12 pixi install`.

**With pip into an existing conda environment**

```sh
conda install -c conda-forge -c ccpi cil tigre
pip install chunkct
```

## Development

Once per clone, set up the git integration:

```sh
pixi run -e cpu nbstripout --install   # stage notebooks without their outputs
pixi run -e cpu prek install           # run ruff and small file checks on every commit
```

After that, `git add` stages notebooks without their outputs (your local copies keep them). To
stage one notebook with its outputs: `git -c filter.nbstripout.clean=cat add <notebook>`.
The checks run on the staged files at `git commit`; if one fails or fixes a file, review the
change, `git add` it again and commit. `pixi run -e cpu prek run --all-files` runs them by hand.

All common commands are pixi tasks:

| Command | What it does |
| --- | --- |
| `pixi run lint` / `pixi run fmt` | ruff lint + format check / autofix |
| `pixi run test` | fast CPU tests (seconds) |
| `pixi run test-gpu` | GPU tests: chunked vs. unchunked FDK equality (needs a GPU) |
| `pixi run hpc-test-gpu` | submit the GPU tests as an LSF job (DTU HPC) |
