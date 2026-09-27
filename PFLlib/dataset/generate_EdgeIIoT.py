import numpy as np
import os
import sys
import random
import shutil
import ujson
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "EdgeIIoT/"

csv_name = "DNN-EdgeIIoT-dataset.csv"
kaggle_handle = "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"
kaggle_file = "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
kaggle_url = "https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"

# Host addresses, timestamps and raw payloads identify the attack run itself, so a
# classifier trained on them scores ~100% without learning anything transferable.
# Same list the dataset authors drop in their reference notebook.
drop_columns = [
    "frame.time", "ip.src_host", "ip.dst_host",
    "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "http.file_data", "http.request.full_uri", "http.request.uri.query",
    "icmp.transmit_timestamp", "tcp.options", "tcp.payload", "mqtt.msg",
]

max_per_class = 5000  # the raw CSV is 2.2M rows and 99% of it is one class
max_categories = 20   # per categorical column, before one-hot
chunk_size = 200000
min_csv_bytes = 200 * 1024 * 1024  # complete file is ~297MB; smaller means a broken download


def find_csv(root):
    if not os.path.exists(root):
        return None
    for dirpath, _, filenames in os.walk(root):
        if csv_name in filenames:
            return os.path.join(dirpath, csv_name)
    return None


def is_complete_csv(path):
    return bool(path) and os.path.isfile(path) and os.path.getsize(path) >= min_csv_bytes


def fetch_csv(raw_dir):
    csv_path = os.path.join(raw_dir, csv_name)
    if is_complete_csv(csv_path):
        return csv_path
    if os.path.exists(csv_path):
        print(f"Incomplete local CSV ({os.path.getsize(csv_path)} bytes); re-downloading.")
        os.remove(csv_path)

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    cache_root = os.path.join(
        os.path.expanduser("~"), ".cache", "kagglehub", "datasets",
        "mohamedamineferrag", "edgeiiotset-cyber-security-dataset-of-iot-iiot",
    )
    cached = find_csv(cache_root)
    if is_complete_csv(cached):
        print(f"Reusing kagglehub cache: {cached}")
        shutil.copy(cached, csv_path)
        return csv_path
    if cached:
        print(f"Incomplete kagglehub cache ({os.path.getsize(cached)} bytes); re-downloading.")
        os.remove(cached)

    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub is not installed. Either run `pip install kagglehub`, or download\n"
            f"  {kaggle_url}\n"
            f"manually and place {csv_name} in {os.path.abspath(raw_dir)}"
        )

    print("\nDownloading Edge-IIoTset from Kaggle (via kagglehub)...")
    try:
        downloaded_path = kagglehub.dataset_download(handle=kaggle_handle, path=kaggle_file)
    except Exception as e:
        raise SystemExit(
            f"Download failed: {e}\n"
            "Kaggle needs credentials for this dataset. Either set KAGGLE_USERNAME/KAGGLE_KEY\n"
            f"(or ~/.kaggle/kaggle.json), or download {kaggle_url}\n"
            f"manually and place {csv_name} in {os.path.abspath(raw_dir)}"
        )

    source = downloaded_path if os.path.isfile(downloaded_path) else find_csv(downloaded_path)
    if source is None:
        raise SystemExit(f"kagglehub finished but {csv_name} was not found under {downloaded_path}")
    if not is_complete_csv(source):
        raise SystemExit(
            f"Downloaded file is incomplete ({os.path.getsize(source) if source and os.path.exists(source) else 0} bytes). "
            "Delete the kagglehub cache and retry."
        )
    shutil.copy(source, csv_path)
    print(f"Dataset downloaded to {csv_path}")
    return csv_path


def load_frame(csv_path, cap):
    """Read the CSV in chunks, keeping at most `cap` rows per attack type."""
    header = pd.read_csv(csv_path, nrows=0)
    usecols = [c for c in header.columns if c not in drop_columns]

    rng = np.random.RandomState(1)
    budget = defaultdict(lambda: cap)
    kept = []

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        chunk = chunk.dropna().drop_duplicates()
        for label, group in chunk.groupby("Attack_type"):
            room = budget[label]
            if room <= 0:
                continue
            if len(group) > room:
                group = group.sample(n=int(room), random_state=rng)
            budget[label] -= len(group)
            kept.append(group)

    df = pd.concat(kept, ignore_index=True)
    return df.sample(frac=1, random_state=1).reset_index(drop=True)


def build_features(df):
    labels = df.pop("Attack_type")
    df = df.drop(columns=["Attack_label"], errors="ignore")

    columns = []
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() > 0.95:
            columns.append(numeric.fillna(0.0).astype(np.float32).rename(col))
        else:
            top = df[col].value_counts().index[:max_categories]
            capped = df[col].where(df[col].isin(top), other="__other__")
            columns.append(pd.get_dummies(capped, prefix=col).astype(np.float32))

    X = pd.concat(columns, axis=1)
    X = X.loc[:, X.std() > 0]
    print(f"Feature columns after encoding: {X.shape[1]}")
    X = StandardScaler().fit_transform(X.to_numpy(np.float32)).astype(np.float32)

    classes = sorted(labels.unique())
    y = labels.map({name: i for i, name in enumerate(classes)}).to_numpy(np.int64)
    return X, y, classes


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition, cap):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    csv_path = fetch_csv(dir_path + "rawdata")
    df = load_frame(csv_path, cap)
    print(f"Rows kept: {len(df)}")

    dataset_features, dataset_label, classes = build_features(df)
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")
    for i, name in enumerate(classes):
        print(f"  {i}: {name} ({int((dataset_label == i).sum())} samples)")

    X, y, statistic = separate_data((dataset_features, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
        statistic, niid, balance, partition)

    # main.py sizes the input layer from these, since it varies with the encoding
    with open(config_path, 'r') as f:
        config = ujson.load(f)
    config['num_features'] = int(dataset_features.shape[1])
    config['classes'] = list(classes)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else max_per_class
    if cap <= 0:
        cap = np.inf

    generate_dataset(dir_path, num_clients, niid, balance, partition, cap)
