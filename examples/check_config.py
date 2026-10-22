"""Check that examples/example.config parses with the project's own configuration loader.

This does not train anything: it needs no dataset, no TensorFlow and no PyTorch. It only
proves that the example file is well formed, that the typed sections are honoured, and that
the keys the training scripts read are present.

Run from the repository root:

    python examples/check_config.py
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

# Load utils/config.py directly instead of importing the package: utils/__init__.py imports
# the TensorFlow-dependent modules, and this check must run without TensorFlow installed.
_config_path = Path(__file__).resolve().parent.parent / "metallograph_segmentation" / "utils" / "config.py"
_spec = spec_from_file_location("metallograph_config", _config_path)
_config = module_from_spec(_spec)
_spec.loader.exec_module(_config)
get_config = _config.get_config

# Keys the training scripts read, taken from a scan of models/sm/train.py and
# models/multiview/train.py.
REQUIRED = (
    "dataset",
    "datadir",
    "modeldir",
    "name",
    "num_classes",
    "kfold",
    "run_cv",
    "run_test",
    "validation_ratio",
    "debug",
)


def main() -> None:
    config_path = Path(__file__).with_name("example.config")
    config = get_config(str(config_path))

    missing = [key for key in REQUIRED if key not in config]
    if missing:
        raise SystemExit(f"example.config is missing: {', '.join(missing)}")

    # The typed sections must actually produce typed values.
    assert isinstance(config["num_classes"], int), "num_classes must come from an [int] section"
    assert isinstance(config["validation_ratio"], float), "validation_ratio must come from a [float] section"
    assert isinstance(config["run_cv"], bool), "run_cv must come from a [bool] section"

    print(f"Parsed {config_path.name} with {len(config)} keys:")
    for key in REQUIRED:
        print(f"  {key:18} = {config[key]!r}")

    print("\nNote: `datadir` must point at a real directory of MetalDAM .h5 files before")
    print("any script can train. This check only validates the configuration file.")


if __name__ == "__main__":
    main()
