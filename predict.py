import argparse
import csv
import os

import numpy as np
import scipy.spatial
import torch

from model import VoteMLP


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            rows.append(row)
    return np.array(rows, dtype=np.float64)


def build_features(data, input_size, feat_mean, feat_std):
    
    coordinates = data[:, 1:3]                              # lon, lat
    base = data[:, 1:5].astype(np.float32)                  # lon, lat, pop, votes
    base_norm = ((base - feat_mean) / feat_std).astype(np.float32)

    n = coordinates.shape[0]
    tree = scipy.spatial.cKDTree(coordinates)
    k = min(input_size, n)
    _, idx = tree.query(coordinates, k=k, workers=-1)       # (N, k)
    if k == 1:
        idx = idx[:, None]
    # pad with the farthest neighbor if the block set is smaller than input_size
    if k < input_size:
        idx = np.concatenate([idx, np.repeat(idx[:, -1:], input_size - k, axis=1)], axis=1)

    return base_norm[idx].reshape(n, input_size * 4)


def predict(model_path, test_path, device=None):
    device = device or get_device()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    input_size = ckpt["input_size"]

    model = VoteMLP(in_dim=input_size * 4).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data = load_csv(test_path)
    X = build_features(data, input_size, ckpt["feat_mean"], ckpt["feat_std"])

    preds = []
    with torch.no_grad():
        for start in range(0, X.shape[0], 65536):
            xb = torch.from_numpy(X[start:start + 65536]).to(device)
            preds.append(model(xb).cpu().numpy())
    return data, np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--model", default="model.pt")
    args = ap.parse_args()

    data, preds = predict(args.model, args.test_csv)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        for i in range(data.shape[0]):
            w.writerow([
                int(data[i, 0]),                            # id
                round(float(data[i, 1]), 7),                # lon
                round(float(data[i, 2]), 7),                # lat
                int(data[i, 3]),                            # voting_pop
                round(float(data[i, 4]), 4),                # total_votes
                round(float(preds[i, 0]), 6),               # r_vote_pred
                round(float(preds[i, 1]), 6),               # d_vote_pred
            ])
    print(f"Wrote {data.shape[0]} predictions -> {args.out_csv}")


if __name__ == "__main__":
    main()
