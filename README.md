# S-metallograph-segmentation

Code for semantic segmentation models and techniques applied to metal microstructures, in
collaboration with ArcelorMittal. It accompanies the *MetalDAM* tutorial paper: the
taxonomy of the problem, the dataset, the deep-learning ensemble and the experimental
analysis.

## What is inside

Each folder under `metallograph_segmentation/models/` is a self-contained model family with
its own training entry point:

| Folder | Approach |
| --- | --- |
| `models/sm` | Segmentation models (U-Net and relatives, via `segmentation-models-pytorch`). |
| `models/multiview` | Multi-view model, trained on several views of the same specimen. |
| `models/stacking` | Stacking of models, including the k-fold variant (`trainK3.py`). |
| `models/usss` | Unsupervised / semi-supervised setup (DRNet). |
| `models/pixelnet` | PixelNet-style pixel-level model. |

Shared code lives in `metallograph_segmentation/utils/`: configuration, data loading,
cross-validation, losses, metrics, prediction and saving.

## Requirements

- Python 3, with **both** PyTorch and TensorFlow available: the models use PyTorch, while
  `utils/data_loader.py` imports `tensorflow.keras` for categorical encoding and image
  loading.
- `h5py` and `scipy`, which `utils/data_loader.py` imports but `setup.py` does not declare.
  Install them explicitly:
  ```bash
  pip install h5py scipy
  ```

## Installation

The package is not on PyPI; install it from a clone:

```bash
git clone https://github.com/ari-dasci/S-metallograph-segmentation.git
cd S-metallograph-segmentation
pip install -e .
```

`setup.py` also registers the training scripts as console scripts. Because four of them are
called `train.py` (and two `trainK3.py`), they collide when installed into the same `bin`
directory; running the modules directly, as shown below, avoids the issue.

## Data

The experiments use the MetalDAM dataset, in HDF5 (`.h5`) format. The loaders in
`utils/data_loader.py` expect one `.h5` file per dataset split, under the directory given by
the `datadir` configuration key (`uhcs.h5`, `arcelor.h5`, and so on, depending on the
`dataset` key). Nothing else is downloaded automatically: point `datadir` at your local copy.

## Running a model

Every training script takes a single argument, the path to an INI configuration file:

```bash
cd metallograph_segmentation
python models/sm/train.py path/to/my.config
```

The configuration is read by `utils/config.py`, which understands the `[general]`, `[str]`,
`[int]`, `[float]` and `[bool]` sections, and is case-insensitive on keys. The keys the
scripts read are:

| Key | Meaning |
| --- | --- |
| `dataset` | Dataset/split family to load (`uhcs`, `arcelor`, ...). |
| `datadir` | Directory holding the `.h5` files. |
| `num_classes` | Number of segmentation classes. |
| `modeldir` / `save_dir` | Where checkpoints and results are written. |
| `name` | Name of the run. |
| `kfold` | Number of cross-validation folds. |
| `run_cv`, `run_test`, `validation_ratio` | Whether to run cross-validation and testing, and the validation split. |
| `kernel_size`, `colors`, `models` | Model-family specific options. |
| `debug` | Verbose output. |

Note that the scripts fall back to a default configuration path
(`models/sm/config/unet_metaldam_pretrined.config` for `models/sm/train.py`), and **no
`.config` file is committed to this repository**. Write your own before running them, or the
script stops at the configuration read.

A starting point is in [`examples/example.config`](examples/example.config), with every key
the scripts read and the typed sections the parser expects. To check that your own file is
well formed, without needing TensorFlow, PyTorch or the dataset:

```bash
python examples/check_config.py
```

## Publication and citation

> [A tutorial on the segmentation of metallographic images: Taxonomy, new MetalDAM dataset, deep learning-based ensemble model, experimental analysis and challenges. Julián Luengo, Raúl Moreno, Iván Sevillano, David Charte, Adrián Peláez-Vegas, Marta Fernández-Moreno, Pablo Mesejo and Francisco Herrera. Information Fusion 78 (232-253), 2022.](https://www.sciencedirect.com/science/article/pii/S1566253521001949)

If you want to cite this Python package, please use the following citation:

~~~latex
@article{metalseg,
  title = {A tutorial on the segmentation of metallographic images: Taxonomy, new MetalDAM dataset, deep learning-based ensemble model, experimental analysis and challenges},
  journal = {Information Fusion},
  volume = {78},
  pages = {232-253},
  year = {2022},
  issn = {1566-2535},
  doi = {https://doi.org/10.1016/j.inffus.2021.09.018},
  url = {https://www.sciencedirect.com/science/article/pii/S1566253521001949},
  author = {Julián Luengo and Raúl Moreno and Iván Sevillano and David Charte and Adrián Peláez-Vegas and Marta Fernández-Moreno and Pablo Mesejo and Francisco Herrera},
}
~~~

## License

GNU General Public License v3.0 — see the [LICENSE](LICENSE) file.
