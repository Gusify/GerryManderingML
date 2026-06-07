import torch
import scipy
import csv

INPUT_SIZE = 100
def main():
    print(f"PyTorch version: {torch.__version__}")
    
    state_names = ["california", "missouri", "montana", "oklahoma", "texas"]
    training_data = []

    for name in state_names:
        # Gus you'll probably need to edit this path unless you copy mine
        file_path = "D:/AIProj1/cleandata/" + name + "_training.csv"
        with open(file_path, 'r', newline='') as csvtestfile:
            reader = csv.reader(csvtestfile)
            for row in reader:
                training_data.append(row)

    print(len(training_data))

    coordinates = []
    for block in training_data:
        x_y = [block[1], block[2]]
        coordinates.append(x_y)
    
    block_tree = scipy.spatial.KDTree(coordinates, leafsize=100)

    for block in coordinates:
        input_data = []
        # scipy idea from top answer on https://stackoverflow.com/questions/12923586/nearest-neighbor-search-python
        result = block_tree.query(block, k=INPUT_SIZE)
        #result[0] is distances, doesn't matter. result[1] has indexes, result[1][0] is self
        nearest_indexes = result[1]
        for index in nearest_indexes:
            input_data.append(training_data[index])

        #do we want to try numpy?
        x = torch.tensor(input_data)
        #Input into model here...
        





if __name__ == "__main__":
    main()
