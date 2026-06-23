import torch
import scipy
import numpy as np
import csv
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import VoteMLP

INPUT_SIZE = 64
NUM_EPOCHS = 100
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
MODEL_OUT = "model.pt"
def main():
    print(f"PyTorch version: {torch.__version__}")
    
    state_names = ["california", "missouri", "montana", "oklahoma", "texas", "tennessee"]
    training_data = []

    for name in state_names:
        # Gus you'll probably need to edit this path unless you copy mine
        file_path = "cleandata/" + name + "_training.csv"
        with open(file_path, 'r', newline='') as csvtestfile:
            reader = csv.reader(csvtestfile, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                training_data.append(row)

    training_data = np.array(training_data)
    training_data = training_data[:, 1:] #keep all rows of data, remove id

    #all rows, just latitude and longitude for coordinates. np array
    coordinates = training_data[:, 0:2]
    
    # scipy idea from top answer on https://stackoverflow.com/questions/12923586/nearest-neighbor-search-python
    block_tree = scipy.spatial.KDTree(coordinates, leafsize=100)

    # holds a list of n nearest neighbors for our inputs. dict key is index, includes self in array
    n_nearest_neighbors = {}
    for block in coordinates:
        result = block_tree.query(block, k=INPUT_SIZE)
        #result[0] is distances, doesn't matter. result[1] has indexes, result[1][0] is self
        nearest_indexes = result[1]
        block = result[1][0]
        n_nearest_neighbors[block] = nearest_indexes


    base = training_data[:, 0:4].astype(np.float32)        # lon, lat, pop, votes
    targets = training_data[:, 4:6].astype(np.float32)     # r_vote, d_vote
    num_blocks = training_data.shape[0]

    feat_mean = base.mean(axis=0)
    feat_std = base.std(axis=0)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    base_norm = ((base - feat_mean) / feat_std).astype(np.float32)

    neighbor_idx = np.empty((num_blocks, INPUT_SIZE), dtype=np.int64)
    for index_counter in range(num_blocks):
        neighbor_idx[index_counter] = n_nearest_neighbors.get(index_counter)

    X = base_norm[neighbor_idx].reshape(num_blocks, INPUT_SIZE * 4)
    Y = targets

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}  |  {num_blocks} blocks, input dim {INPUT_SIZE * 4}")

    # train / validation split
    rng = np.random.default_rng(0)
    perm = rng.permutation(num_blocks)
    X, Y = X[perm], Y[perm]
    n_val = int(num_blocks * 0.1)
    Xtr, Ytr = X[n_val:], Y[n_val:]
    Xva = torch.from_numpy(X[:n_val]).to(device)
    Yva = torch.from_numpy(Y[:n_val]).to(device)

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    model = VoteMLP(in_dim=INPUT_SIZE * 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.L1Loss() #Mean Absolute Error

    for epoch in range(NUM_EPOCHS):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.shape[0]

        model.eval()
        with torch.no_grad():
            vpred = model(Xva)
            val_mae = (vpred - Yva).abs().mean().item()
            win_acc = ((vpred[:, 0] > vpred[:, 1]) == (Yva[:, 0] > Yva[:, 1])).float().mean().item()

        print(f"epoch {epoch + 1:>3} val_mae {val_mae:.5f}  win_acc {win_acc:.3f}")

    torch.save(
        {"state_dict": model.state_dict(), "input_size": INPUT_SIZE,
         "feat_mean": feat_mean, "feat_std": feat_std},
        MODEL_OUT,
    )
    print(f"Saved model -> {MODEL_OUT}")

            




if __name__ == "__main__":
    main()
