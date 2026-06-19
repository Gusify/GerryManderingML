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
    #ap = argparse.ArgumentParser()
    #ap.add_argument("predict_csv") # where to get data from
    #ap.add_argument("districts_csv") # where to print district mappings to
    #args = ap.parse_args()

    predicted_data = load_csv("D:/AIProj1/predict_data/california_predict.csv")
    num_districts = 12
    democrat_districts = 0 #treat republican as num_districts - democrat_districts, 2 variables seems less good
    for data in predicted_data:
        data[3] = round(data[4] * data[5]) #r_vote * num_votes
        data[4] = round(data[4] * data[6]) #d_vote * num_votes
    predicted_data = np.delete(predicted_data, 5, axis=1) # trim extra columns
    predicted_data = np.delete(predicted_data, 5, axis=1)
    #new shape is [id, long, lat, republican_votes, democrat_votes]
    #worth noting btw that the total is often 1-2 less than the expected vote amount because of rounding. Could do ceiling and have it be more also
    
    





if __name__ == "__main__":
    main()
