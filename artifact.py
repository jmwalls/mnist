"""XXX
"""
import gzip
from pathlib import Path
import pickle
import requests

import tqdm
import wandb


ARTIFACT = "mnist"
TRAIN_NAME = "train.pkl.gz"
VALIDATION_NAME = "validation.pkl.gz"


def download_mnist(path: Path, *, force_download: bool = False) -> Path:
    """
    Download gzipped MNIST data.

    @param path: path to output diretory
    @param force_download: force download even if file already exists

    @returns path to gzipped MNIST data
    """
    url = "https://github.com/pytorch/tutorials/raw/main/_static/"
    fname = "mnist.pkl.gz"

    out = path / fname
    if not out.exists() or force_download:
        print("downloading MNIST...")
        content = requests.get(url + fname).content
        out.open("wb").write(content)
        print("done")
    return out


def main():
    datadir = Path("data-cache")
    datadir.mkdir(exist_ok=True)

    path = download_mnist(datadir)
    with gzip.open(path, "rb") as f:
        ((x_train, y_train), (x_val, y_val), _) = pickle.load(f, encoding="latin-1")

    run = wandb.init(project="mnist", entity=None, job_type="upload")
    dataset = wandb.Artifact(ARTIFACT, type="dataset")

    # Add train/validation to artifact.
    print("adding train/val to artifact...")
    with gzip.open(datadir / TRAIN_NAME, "wb") as f:
        pickle.dump((x_train, y_train), f)
    dataset.add_file(datadir / TRAIN_NAME, name=TRAIN_NAME)

    with gzip.open(datadir / VALIDATION_NAME, "wb") as f:
        pickle.dump((x_val, y_val), f)
    dataset.add_file(datadir / VALIDATION_NAME, name=VALIDATION_NAME)

    # Add summary table to artifact.
    table = wandb.Table(columns=["index", "split", "label", "Image"])

    def _add_to_table(X, Y, split):
        for i, (x, y) in tqdm.tqdm(enumerate(zip(X, Y))):
            table.add_data(i, split, y, wandb.Image(x.reshape(1, 28, 28)))

    print("adding summary table to artifact...")
    _add_to_table(x_train, y_train, "train")
    _add_to_table(x_val, y_val, "validation")
    dataset.add(table, "summary_table")

    run.log_artifact(dataset)
    run.finish()


if __name__ == "__main__":
    main()
