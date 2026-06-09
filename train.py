import torch
import scipy
import numpy as np
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

    training_data = np.array(training_data)
    training_data = training_data[:, 1:] #remove id

    coordinates = []
    for block in training_data:
        long_lat = [block[0], block[1]]
        coordinates.append(long_lat)
    
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

    for epoch in range(NUM_EPOCHS):
        index_counter = 0
        for row in training_data:
            x = [] #row is first thing in nearest neighbors, don't need to add
            y = row[4:] # R_vote, D_vote
            nearest_neighbors = n_nearest_neighbors.get(index_counter)
            for neighbor in nearest_neighbors:
                x.append(training_data[neighbor][0:4])
            index_counter += 1 

            




if __name__ == "__main__":
    main()
