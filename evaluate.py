import argparse

import numpy as np

from predict import load_csv, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_csv")
    ap.add_argument("truth_csv")
    ap.add_argument("--model", default="model.pt")
    args = ap.parse_args()

    data, preds = predict(args.model, args.test_csv)
    truth = load_csv(args.truth_csv)                        # id..d_vote,r_vote

    truth_by_id = {int(r[0]): (r[5], r[6]) for r in truth}
    y = np.array([truth_by_id[int(i)] for i in data[:, 0]], dtype=np.float32)

    mae = np.abs(preds - y).mean()
    mae_d = np.abs(preds[:, 0] - y[:, 0]).mean()
    mae_r = np.abs(preds[:, 1] - y[:, 1]).mean()
    rmse = np.sqrt(((preds - y) ** 2).mean())
    win_acc = ((preds[:, 0] > preds[:, 1]) == (y[:, 0] > y[:, 1])).mean()

    print(f"blocks      {data.shape[0]}")
    print(f"MAE         {mae:.4f}   (d {mae_d:.4f}, r {mae_r:.4f})")
    print(f"RMSE        {rmse:.4f}")
    print(f"winner acc  {win_acc:.3f}")


if __name__ == "__main__":
    main()
