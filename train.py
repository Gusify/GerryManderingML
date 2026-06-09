import torch
import scipy
import csv

INPUT_SIZE = 64
NUM_EPOCHS = 1
def main():
    print(f"PyTorch version: {torch.__version__}")
    
    state_names = ["california", "missouri", "montana", "oklahoma", "texas"]
    training_data = []

    for name in state_names:
        # Gus you'll probably need to edit this path unless you copy mine
        file_path = "D:/AIProj1/cleandata/" + name + "_training.csv"
        with open(file_path, 'r', newline='') as csvtestfile:
            reader = csv.reader(csvtestfile, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                training_data.append(row)

    print(len(training_data))

    coordinates = []
    for block in training_data:
        x_y = [block[1], block[2]]
        coordinates.append(x_y)
    
    block_tree = scipy.spatial.KDTree(coordinates, leafsize=100)

    # holds a list of n nearest neighbors for our inputs. dict key is index, includes self in array
    n_nearest_neighbors = {}
    for block in coordinates:
        # scipy idea from top answer on https://stackoverflow.com/questions/12923586/nearest-neighbor-search-python
        result = block_tree.query(block, k=INPUT_SIZE)
        #result[0] is distances, doesn't matter. result[1] has indexes, result[1][0] is self
        nearest_indexes = result[1]
        block = result[1][0]
        n_nearest_neighbors[block] = nearest_indexes





if __name__ == "__main__":
    main()
