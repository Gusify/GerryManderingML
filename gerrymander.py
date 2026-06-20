import numpy as np
import pygad
import argparse
import csv


ACCEPTABLE_POPULATION_RANGE = 1.2 # acceptable ratio between state's min:max population district. i.e. 1.2 means 1.2 * min >= max
#taken from predict.py
def load_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            rows.append(row)
    return np.array(rows, dtype=np.float64)

def determine_republican_districts(districts):
    num_r_districts = 0
    for district in districts:
        sum_values = np.sum(district, axis=0)# 0-3 id, lat, lon, pop. 4=R, 5=D
        if sum_values[4] > sum_values[5]: #R_vote > D_vote
            num_r_districts += 1
    
    return num_r_districts


def get_population_per_district(districts):
    populations = []
    for district in districts:
        sum_values = np.sum(district, axis=0) # 3 = pop
        populations.append(sum_values[3])
    populations_np = np.array(populations)
    return populations_np
    
#def generate_points(lon_max, long_min, lat_max, lat_min):


#def fitness_func(ga_instance, solution, solution_idx):


def main():
    #ap = argparse.ArgumentParser()
    #ap.add_argument("predict_csv") # where to get data from
    #ap.add_argument("districts_csv") # where to print district mappings to
    #args = ap.parse_args()

    predicted_data = load_csv("D:/AIProj1/predict_data/tennessee_predict.csv")
    num_districts = 9
    republican_districts = 6 #treat democrat as num_districts - republican_districts, 2 variables seems less good

    for data in predicted_data:
        data[5] = round(data[4] * data[6]) #d_vote * num_votes, done first because 4 is deleted
        data[4] = round(data[4] * data[5]) #r_vote * num_votes,
    predicted_data = np.delete(predicted_data, 6, axis=1) # trim off num votes column
    #new shape is [id, long, lat, pop, republican_votes, democrat_votes]
    #worth noting btw that the total is often 1-2 less than the expected vote amount because of rounding. Could do ceiling and have it be more also
    
    longitudes = predicted_data[:, 1]
    latitudes = predicted_data[:, 2]
    max_longitude = longitudes.max()
    min_longitude = longitudes.min()
    max_latitude = latitudes.max()
    min_latitude = latitudes.min() 

    

    





if __name__ == "__main__":
    main()
