import csv

train_path = "D:/AIProj1/cleandata/tennessee_training.csv"
predict_path = "D:/AIProj1/predict_data/tennessee_predict.csv"
training_data = []
prediction_data = []
with open(train_path, 'r', newline='') as csvtrainfile:
    reader = csv.reader(csvtrainfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        training_data.append(row)

with open(predict_path, 'r', newline='') as csvpredictfile:
    reader = csv.reader(csvpredictfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        prediction_data.append(row)

r_vote_difference = 0
d_vote_difference = 0
for x in range (0, len(training_data)):
    r_vote_difference += abs(training_data[x][5] - prediction_data[x][5])
    d_vote_difference += abs(training_data[x][6] - prediction_data[x][6])

print ("average Republican vote difference: " + str(r_vote_difference / len(training_data)))
print ("average Democrat vote difference: " + str(d_vote_difference / len(training_data)))

