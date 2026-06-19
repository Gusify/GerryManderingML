import numpy as np
import pygad
import argparse
import csv

#taken from predict.py
def load_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            rows.append(row)
    return np.array(rows, dtype=np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predict_csv") # where to get data from
    ap.add_argument("districts_csv") # where to print district mappings to
    args = ap.parse_args()

    predicted_data = load_csv(args.predict_csv)
    num_districts = 12
    democrat_districts = 0 #treat republican as num_districts - democrat_districts, 2 variables seems less good
    





if __name__ == "__main__":
    main()
