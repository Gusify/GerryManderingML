import numpy as np
import pygad
import argparse
import csv
import random
import math


ACCEPTABLE_POPULATION_RANGE = 1.2 # acceptable ratio between state's min:max population district. i.e. 1.2 means 1.2 * min >= max
STATE = "tennessee" #for filepath
NUM_DISTRICTS = 9
REPUBLICAN_DISTRICTS = 6 #treat democrat as num_districts - republican_districts, 2 variables seems less good
#taken from predict.py
def load_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            rows.append(row)
    return np.array(rows, dtype=np.float64)

def to_district_array(blocks):
    districts = [[] for _ in range (NUM_DISTRICTS)]
    for block in blocks:
        district_num = int(block[6] - 1)
        districts[district_num].append(block)

    return districts


def determine_republican_districts(districts): # gets # republican districts. Treat democrat as n-return value
    num_r_districts = 0
    for district in districts:
        sum_values = np.sum(district, axis=0)# 0-3 id, lat, lon, pop. 4=R, 5=D
        if sum_values[4] > sum_values[5]: #R_vote > D_vote
            num_r_districts += 1
    return num_r_districts


def get_population_per_district(districts): # array of population per district
    populations = []
    for district in districts:
        sum_values = np.sum(district, axis=0) # 3 = pop
        populations.append(sum_values[3])
    populations_np = np.array(populations)
    return populations_np
    

 
def _equal_cuts(pops, n):
    """Return n+1 cut indices into pops such that each chunk has
    roughly equal total population."""
    cum = np.cumsum(pops)
    total = cum[-1]
    cuts = [0]
    for k in range(1, n):
        cuts.append(int(np.searchsorted(cum, total * k / n)))
    cuts.append(len(pops))
    return cuts
 
 
def equal_population_districts(blocks, num_districts=NUM_DISTRICTS,
                                rotation_deg=0, num_strips=None):
    """
    Assigns a district number (0-indexed) to each block in block[6].
 
    Approach:
      1. Rotate lon/lat by rotation_deg, then sort blocks by rotated latitude
         to form horizontal strips at 0 deg, vertical at 90 deg, etc.
      2. Divide blocks into num_strips strips of equal population.
      3. Within each strip, sort by rotated longitude (alternating direction
         per strip for a snake pattern so adjacent strips line up).
      4. Divide each strip into equal-population districts and label blocks.
 
    rotation_deg : controls strip orientation -- randomize per GA individual.
    num_strips   : defaults to round(sqrt(num_districts)).
    """
    # --- rotate ---
    theta = math.radians(rotation_deg)
    cos_a, sin_a = math.cos(theta), math.sin(theta)
    def rotate(lon, lat):
        return lon * cos_a - lat * sin_a, lon * sin_a + lat * cos_a
 
    # --- sort by rotated latitude ---
    blocks = sorted(blocks, key=lambda b: rotate(b[1], b[2])[1])
    rot = [rotate(b[1], b[2]) for b in blocks]
 
    # --- split into strips ---
    num_strips = num_strips or max(1, round(num_districts ** 0.5))
    base, rem = divmod(num_districts, num_strips)
    districts_per_strip = [base + (1 if i < rem else 0) for i in range(num_strips)]
 
    pops = np.array([b[3] for b in blocks], dtype=float)
    strip_cuts = _equal_cuts(pops, num_strips)
 
    # --- assign district numbers ---
    district_num = 0
    for s, d_count in enumerate(districts_per_strip):
        lo, hi = strip_cuts[s], strip_cuts[s + 1]
        strip_blocks = blocks[lo:hi]
        strip_rot = rot[lo:hi]
 
        # snake: reverse every other strip so districts line up across strips
        reverse = (s % 2 == 1)
        strip_blocks = sorted(strip_blocks, key=lambda b: rotate(b[1], b[2])[0], reverse=reverse)
        strip_rot = [rotate(b[1], b[2]) for b in strip_blocks]
 
        strip_pops = np.array([b[3] for b in strip_blocks], dtype=float)
        d_cuts = _equal_cuts(strip_pops, d_count)
 
        for d in range(d_count):
            for block in strip_blocks[d_cuts[d]:d_cuts[d + 1]]:
                block[6] = district_num
            district_num += 1
 
    return blocks


def get_district_centroids(districts): # centers of all districts[[lon, lat] ...]
    centroids = []
    for district in districts:
        lons = district[:, 1]
        lats = district[:, 2]
        center_lon = (max(lons) + min(lons)) / 2
        center_lat = (max(lats) + min(lats)) / 2
        centroids.append([center_lon, center_lat])
    centroids_np = np.array(centroids)
    return centroids_np


        

def fitness_func(ga_instance, solution, solution_idx): #rewards closer seats and more population spread without being over limit
    seat_difference = abs(determine_republican_districts(solution) - REPUBLICAN_DISTRICTS)
    district_main_value = NUM_DISTRICTS - seat_difference # higher number when seats closer to desired
    pops = get_population_per_district(solution)
    population_ratio = max(pops) / min(pops)
    if population_ratio <= ACCEPTABLE_POPULATION_RANGE: #good maps tend to have some disparity in population but not too much. Helps to encourage some changes but less than a seat's worth
        district_main_value += (population_ratio - 1) # will add value. cannot be <1
    else:
        district_main_value -= (population_ratio - 1) # seats matter more but still punish incorrect population distribution
    return district_main_value


def main():
    #ap = argparse.ArgumentParser()
    #ap.add_argument("predict_csv") # where to get data from
    #ap.add_argument("districts_csv") # where to print district mappings to
    #args = ap.parse_args()

    predicted_data = load_csv("D:/AIProj1/predict_data/" + STATE +"_predict.csv")


    for data in predicted_data:
        temp = round(data[4] * data[6]) #d_vote * num_votes, assigned later to not impact data
        data[4] = round(data[4] * data[5]) #r_vote * num_votes,
        data[5] = temp
    #new shape is [id, long, lat, pop, republican_votes, democrat_votes, bad column filled in later]
    #worth noting btw that the total is often 1-2 less than the expected vote amount because of rounding. Could do ceiling and have it be more also
    
    sorted_blocks = equal_population_districts(predicted_data)
    districts = to_district_array(sorted_blocks)
    sorted_blocks_2 = equal_population_districts(predicted_data, num_strips= 2, rotation_deg=63)
    districts_2 = to_district_array(sorted_blocks_2)
    print(fitness_func(None, districts, None))
    print(determine_republican_districts(districts))
    print(fitness_func(None, districts_2, None))
    print(determine_republican_districts(districts_2))

    
    

    





if __name__ == "__main__":
    main()
