# Run the CIFAR experiment

From the repository root, install the packages in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r script/stage_3_script/requirements_cifar.txt
```

Place the course-provided `CIFAR` pickle at `data/stage_3_data/CIFAR`, or pass `--data /path/to/CIFAR`. Load only trusted pickle files.

Evaluate the included validation-selected checkpoint:

```bash
python script/stage_3_script/script_cnn_cifar.py --test
```

Train a fresh run in a new output folder:

```bash
python script/stage_3_script/script_cnn_cifar.py --out result/stage_3_result/CIFAR/new_run
```

The default proposed output already contains the completed run and is protected from accidental training overwrites. The script chooses CUDA, Apple MPS, or CPU automatically; `--device cpu` overrides this.

Available comparison variants are `no_augmentation`, `no_batchnorm`, `no_dropout`, and `shallower`. Use `--variant NAME` with a new `--out` folder for each. These comparison runs have not been performed. Select configurations using validation scores rather than repeated test evaluation.

The architecture lives in `local_code/stage_3_code/Method_CNN_CIFAR.py`. Training uses the supplied 50,000-image training set, split into 45,000 training and 5,000 validation images, with seed 21. Defaults are 40 epochs and batches of 128. The supplied 10,000-image test set is used only with `--test`.

Completed outputs are in `result/stage_3_result/CIFAR/proposed/`: model weights, configuration, exact split indices, epoch history, best validation scores, final test metrics, runtime, loss plot, and confusion matrix (rows are true labels; columns are predicted labels). Metrics are fractions; precision, recall, and F1 use macro averaging. Numeric label order is 0 through 9. The dataset and report documents are not included.
