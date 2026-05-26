import shapefile
import csv

# uncomment and set your own paths. Make sure to not include file extension
# sf_path = "rawdata/tx_2020_gen_2020_blocks/tx_2020_gen_2020_blocks"
# csv_train_path = "cleandata/texas_training.csv"
# csv_test_path = "cleandata/texas_testing.csv"

sf = shapefile.Reader(sf_path)
fields = sf.fields[1:]
field_names = [field[0] for field in fields]
democrat_fields = []
republican_fields = []
field_counter = 0
for field in field_names:
    if field[6:7] == "D":
        democrat_fields.append(field_counter)
    if field[6:7] == "R":
        republican_fields.append(field_counter)
    field_counter += 1

records = sf.shapeRecords()
train_data = []
test_data = []
for data in records:
    lat = round((data.shape.bbox[0] + data.shape.bbox[2]) / 2, 7) # 7 because Nix's coords were at 7 decimal places
    long = round((data.shape.bbox[1] + data.shape.bbox[3]) / 2, 7)
    id = data.record[0]
    voting_pop = data.record[4]
    d_vote = 0
    r_vote = 0
    total_votes = 0
    index = 5
    for votes in data.record[5:]:
        if index in democrat_fields:
            d_vote += votes
        if index in republican_fields:
            r_vote += votes
        index += 1
        total_votes += votes
    if total_votes > 0:
        d_vote = d_vote / total_votes
        r_vote = r_vote / total_votes
    train_data.append([id, lat, long, voting_pop, d_vote, r_vote])
    test_data.append([id, lat, long, voting_pop])
    
with open(csv_train_path, 'w', newline='') as csvtrainfile:
    writer = csv.writer(csvtrainfile)
    writer.writerows(train_data)

with open(csv_test_path, 'w', newline='') as csvtestfile:
    writer = csv.writer(csvtestfile)
    writer.writerows(test_data)

