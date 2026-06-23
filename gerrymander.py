import numpy as np
import pygad
import argparse
import csv
import random
import math
import os
import heapq
import scipy
import scipy.spatial


ACCEPTABLE_POPULATION_RANGE = 1.15 # acceptable ratio between state's min:max population district. i.e. 1.2 means 1.2 * min >= max
STATE = "california" #for filepath, feel free to change to argparse
NUM_DISTRICTS = 52
REPUBLICAN_DISTRICTS = 10 #treat democrat as num_districts - republican_districts, 2 variables seems less good
ADJACENCY = None #block adjacency list (Delaunay), index-aligned with the genome. Set in main
NUM_BLOCKS = None #number of blocks, set in main
#taken from predict.py
def load_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            rows.append(row)
    return np.array(rows, dtype=np.float64)


def build_adjacency(coordinates):
    # We only have block centroids (points), not polygons, so approximate
    # "neighboring blocks" with a Delaunay triangulation of the centroids.
    # Returns an adjacency list aligned with the genome's block ordering.
    coords = np.ascontiguousarray(coordinates, dtype=np.float64)
    try:
        tri = scipy.spatial.Delaunay(coords)
    except scipy.spatial.QhullError:
        # joggle as a fallback only for degenerate (e.g. collinear) point sets;
        # QJ can fail on large well-formed inputs, so it is not the default
        tri = scipy.spatial.Delaunay(coords, qhull_options="QJ")
    adj = [set() for _ in range(len(coords))]
    for a, b, c in tri.simplices:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return [np.fromiter(s, dtype=np.int64) for s in adj]


def grow_districts(coordinates, populations, adjacency, num_districts, rng):
    # Build a contiguous, roughly equal-population partition by growing districts
    # outward on the adjacency graph. A block is only ever assigned when it borders
    # the district claiming it, so every district is guaranteed connected. We always
    # grow the currently-smallest district, which keeps populations balanced.
    n = len(coordinates)
    assign = np.full(n, -1, dtype=np.int64)

    # spread the seeds out with farthest-point sampling (random first seed for variety)
    seeds = [int(rng.integers(n))]
    d2 = np.sum((coordinates - coordinates[seeds[0]]) ** 2, axis=1)
    for _ in range(num_districts - 1):
        nxt = int(np.argmax(d2))
        seeds.append(nxt)
        d2 = np.minimum(d2, np.sum((coordinates - coordinates[nxt]) ** 2, axis=1))

    dist_pop = np.zeros(num_districts)
    frontier = [set() for _ in range(num_districts)]
    for d, s in enumerate(seeds):
        assign[s] = d
        dist_pop[d] = populations[s]
        for nb in adjacency[s]:
            if assign[nb] == -1:
                frontier[d].add(int(nb))

    assigned = num_districts
    heap = [(dist_pop[d], d) for d in range(num_districts)]
    heapq.heapify(heap)
    while assigned < n and heap:
        _, d = heapq.heappop(heap)
        block = None
        while frontier[d]:                 # find an unassigned block on this district's border
            cand = frontier[d].pop()
            if assign[cand] == -1:
                block = cand
                break
        if block is None:
            continue                       # district is sealed off; drop it from the heap
        assign[block] = d
        dist_pop[d] += populations[block]
        assigned += 1
        for nb in adjacency[block]:
            if assign[nb] == -1:
                frontier[d].add(int(nb))
        heapq.heappush(heap, (dist_pop[d], d))

    # safety net: fold any stragglers into an adjacent district (still contiguous)
    while assigned < n:
        progressed = False
        for b in np.where(assign == -1)[0]:
            for nb in adjacency[b]:
                if assign[nb] != -1:
                    assign[b] = assign[nb]
                    assigned += 1
                    progressed = True
                    break
        if not progressed:
            break
    return assign


def count_extra_components(assign):
    # 0 means every district is a single connected component (fully contiguous).
    extra = 0
    for d in range(NUM_DISTRICTS):
        members = np.where(assign == d)[0]
        if len(members) == 0:
            continue
        member_set = set(members.tolist())
        seen = set()
        components = 0
        for start in members:
            if start in seen:
                continue
            components += 1
            stack = [int(start)]
            seen.add(int(start))
            while stack:
                node = stack.pop()
                for nb in ADJACENCY[node]:
                    if nb in member_set and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        extra += components - 1
    return extra


def to_district_array(blocks):
    districts = [[] for _ in range (NUM_DISTRICTS)]
    for block in blocks:
        district_num = int(block[6])
        districts[district_num].append(block)
    districts_np = np.array(districts, dtype=object) #dtype - object allows inequal subarray lengths
    return districts_np


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
    

 
def _equal_cuts(pops, n): #helper function for equal_population_districts, generated by Claude Sonnet 4.6
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
                                rotation_deg=0, num_strips=None):#generated by Claude Sonnet 4.6
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
    blocks.sort(key=lambda x: x[0]) # sort by ID to ensure same ordering of list
    blocks_np = np.array(blocks)
    return blocks_np


def get_district_centroids(districts): # centers of all districts[[lon, lat] ...]
    centroids = []
    for district in districts:
        if len(district) == 0:
            centroids.append([0,0]) #prevents empty index error. empty districts are punished by fitness function
            continue
        district = np.array(district)
        lons = district[:, 1]
        lats = district[:, 2]
        center_lon = (max(lons) + min(lons)) / 2
        center_lat = (max(lats) + min(lats)) / 2
        centroids.append([center_lon, center_lat])
    centroids_np = np.array(centroids)
    return centroids_np

        

def fitness_func(ga_instance, solution, solution_idx): #rewards closer seats and more population spread without being over limit
    reshaped_solution = solution.reshape(NUM_BLOCKS, 7)
    districts = to_district_array(reshaped_solution)
    for district in districts:
        if len(district) == 0:
            return -1 #harshly punish empty districts. Also prevents null checks
    seat_difference = abs(determine_republican_districts(districts) - REPUBLICAN_DISTRICTS)
    district_main_value = NUM_DISTRICTS - seat_difference # higher number when seats closer to desired
    pops = get_population_per_district(districts)
    population_ratio = max(pops) / min(pops)
    if population_ratio <= ACCEPTABLE_POPULATION_RANGE: #good maps tend to have some disparity in population but not too much. Helps to encourage some changes but less than a seat's worth
        district_main_value += (population_ratio - 1) # will add value. cannot be <1
    else:
        district_main_value -= (population_ratio - 1) # seats matter more but still punish incorrect population distribution
    # contiguity backstop: 0 in normal operation (seeds + mutation keep maps contiguous),
    # but punishes any non-contiguous district by more than a seat so contiguity always wins
    extra_components = count_extra_components(reshaped_solution[:, 6].astype(np.int64))
    district_main_value -= 2 * extra_components
    return district_main_value

MUTATIONS_PER_INDIVIDUAL = 30 #number of boundary blocks to try flipping per offspring


def _removal_keeps_connected(b, old_d, assign):
    # True if district old_d stays connected after block b is removed from it.
    same = [int(n) for n in ADJACENCY[b] if assign[n] == old_d]
    if len(same) <= 1:
        return True #b is a leaf of its district: removing it can't disconnect it
    target = set(same)
    start = next(iter(target))
    seen = {start}
    stack = [start]
    while stack: #BFS through old_d (skipping b); can we still reach all of b's neighbors?
        node = stack.pop()
        target.discard(node)
        if not target:
            return True
        for nb in ADJACENCY[node]:
            if nb != b and assign[nb] == old_d and nb not in seen:
                seen.add(int(nb))
                stack.append(int(nb))
    return not target


def mutation_func(offspring, ga_instance):
    # Contiguity-preserving mutation: only flip a block into a district it physically
    # borders, and only if that doesn't split or empty its old district. Starting from
    # contiguous seeds this keeps every map contiguous (the AlphaPhoenix-style move).
    return_offspring = []
    for solution in offspring:
        reshaped_offspring = solution.reshape(NUM_BLOCKS, 7).copy()
        assign = reshaped_offspring[:, 6].astype(np.int64)
        sizes = np.bincount(assign, minlength=NUM_DISTRICTS)
        for _ in range(MUTATIONS_PER_INDIVIDUAL):
            b = random.randrange(NUM_BLOCKS)
            old_d = int(assign[b])
            if sizes[old_d] <= 1: #don't empty a district
                continue
            neighbor_districts = {int(assign[n]) for n in ADJACENCY[b]} - {old_d}
            if not neighbor_districts: #interior block, nothing to flip to
                continue
            if not _removal_keeps_connected(b, old_d, assign):
                continue #flip would split the old district
            new_d = random.choice(tuple(neighbor_districts))
            assign[b] = new_d
            sizes[old_d] -= 1
            sizes[new_d] += 1
        reshaped_offspring[:, 6] = assign
        return_offspring.append(reshaped_offspring.flatten())
    return np.array(return_offspring, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predict_csv") # labeled blocks to district (predictions or actual data)
    ap.add_argument("districts_csv") # where to print district mappings to
    ap.add_argument("--num-districts", type=int, default=52) # total number of districts required
    ap.add_argument("--rep-districts", type=int, default=10) # desired Republican districts (Dem = total - this)
    ap.add_argument("--pop-range", type=float, default=1.15) # acceptable max:min district population ratio
    args = ap.parse_args()

    global NUM_DISTRICTS, REPUBLICAN_DISTRICTS, ACCEPTABLE_POPULATION_RANGE
    NUM_DISTRICTS = args.num_districts
    REPUBLICAN_DISTRICTS = args.rep_districts
    ACCEPTABLE_POPULATION_RANGE = args.pop_range

    predicted_data = load_csv(args.predict_csv)

    # input columns: [id, lon, lat, voting_pop, total_votes, d_vote, r_vote]
    # convert the vote proportions into raw R/D vote counts for the fitness function.
    # NOTE: the model writes d_vote in col5 and r_vote in col6 (verified: R-leaning
    # blocks carry the larger value in col6), so R = total_votes*col6, D = total_votes*col5.
    for data in predicted_data:
        r_votes = round(data[4] * data[6]) #total_votes * r_vote
        d_votes = round(data[4] * data[5]) #total_votes * d_vote
        data[4] = r_votes #col4 -> Republican votes
        data[5] = d_votes #col5 -> Democrat votes
    #new shape is [id, long, lat, pop, republican_votes, democrat_votes, bad column filled in later]
    #worth noting btw that the total is often 1-2 less than the expected vote amount because of rounding. Could do ceiling and have it be more also

    # sort blocks by id so every individual shares the same block ordering
    predicted_data = predicted_data[predicted_data[:, 0].argsort()]
    global NUM_BLOCKS
    NUM_BLOCKS = len(predicted_data)
    coordinates = predicted_data[:, 1:3]
    populations = predicted_data[:, 3]
    global ADJACENCY
    ADJACENCY = build_adjacency(coordinates)

    # Seed with contiguous, roughly equal-population maps grown on the adjacency graph.
    # Growing only into adjacent blocks guarantees each district is connected; the GA
    # then shifts borders to hit the desired party split while the contiguity-preserving
    # mutation keeps every map contiguous.
    seed_rng = np.random.default_rng(0)
    initial_population = []
    for _ in range(4):
        assign = grow_districts(coordinates, populations, ADJACENCY, NUM_DISTRICTS, seed_rng)
        individual = predicted_data.copy()
        individual[:, 6] = assign
        initial_population.append(individual.flatten())

    fitness_function = fitness_func
    num_generations = 10
    num_parents_mating = 2

    stop_criteria = "reach_" + str(NUM_DISTRICTS) #stops if fitness function returns value >= num districts, ideal # of seats with a valid population

    num_genes = NUM_BLOCKS * 7

    parent_selection_type = "sss"

    crossover_type = None # contiguity-preserving mutation only; two_points crossover would scatter districts and break contiguity
    mutation_function = mutation_func

    #setup and variables inspired by https://pygad.readthedocs.io/en/latest/index.html
    ga_instance = pygad.GA(num_generations=num_generations, 
                           fitness_func=fitness_function,
                           num_parents_mating=num_parents_mating,
                           num_genes=num_genes,
                           crossover_type=crossover_type,
                           mutation_type=mutation_function,
                           parent_selection_type=parent_selection_type,
                           initial_population=initial_population,
                           stop_criteria=stop_criteria
                           )
    
    ga_instance.run()
    
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    solution = solution.reshape(NUM_BLOCKS, 7)
    solution_districts = to_district_array(solution)
    num_republican_solution = determine_republican_districts(solution_districts)
    populations_solution = get_population_per_district(solution_districts)
    solution_assign = solution[:, 6].astype(np.int64)
    extra_components = count_extra_components(solution_assign)
    print("Number of Districts: " + str(NUM_DISTRICTS))
    print("Desired number of Republican/Democrat Districts: " + str(REPUBLICAN_DISTRICTS)+ "/" + str(NUM_DISTRICTS - REPUBLICAN_DISTRICTS))
    print("Genetic Algorithm Result:" + str(num_republican_solution) + "/" + str(NUM_DISTRICTS - num_republican_solution))
    print("Population per district" + str(get_population_per_district(solution_districts)))
    print("All districts contiguous: " + str(extra_components == 0) + " (extra components: " + str(extra_components) + ")")

    print_file_path = args.districts_csv
    out_dir = os.path.dirname(print_file_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    #Will output all census blocks with the district as the last column
    # with open(print_file_path, "w", newline="") as output_file: 
    #     writer = csv.writer(output_file)
    #     writer.writerow("Number of Districts: " + str(NUM_DISTRICTS))
    #     writer.writerow("Desired number of Republican/Democrat Districts: " + str(REPUBLICAN_DISTRICTS)+ "/" + str(NUM_DISTRICTS - REPUBLICAN_DISTRICTS))
    #     writer.writerow("Genetic Algorithm Result:" + str(num_republican_solution) + "/" + str(NUM_DISTRICTS - num_republican_solution))
    #     writer.writerow(["id", "longitude", "latitude", "voting population", "R votes", "D votes", "district"])
    #     writer.writerows(solution)

    with open(print_file_path, "w", newline="") as output_file: 

        writer = csv.writer(output_file)
        writer.writerow(["Number of Districts: " + str(NUM_DISTRICTS)])
        writer.writerow(["Desired number of Republican/Democrat Districts: " + str(REPUBLICAN_DISTRICTS)+ "/" + str(NUM_DISTRICTS - REPUBLICAN_DISTRICTS)])
        writer.writerow(["Genetic Algorithm Result:" + str(num_republican_solution) + "/" + str(NUM_DISTRICTS - num_republican_solution)])
        district_num = 1
        for district in solution_districts:
            d_votes = 0
            r_votes = 0
            for block in district:
                d_votes += block[5]
                r_votes += block[4]
            writer.writerow(["district num", "population", "R_votes", "D_votes"])
            writer.writerow([str(district_num), populations_solution[district_num - 1], str(r_votes), str(d_votes)])
            writer.writerow(["id", "longitude", "latitude", "voting population", "R votes", "D votes", "district"])
            writer.writerows(district)


    





if __name__ == "__main__":
    main()
