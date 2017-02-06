from sklearn.model_selection import train_test_split
import csv
import numpy as np
from sklearn.utils import shuffle



def tts(data):
    train, test = train_test_split(data, test_size=0.20, random_state=0)

    return train, test



with open('driving_log.csv', 'r') as f:
    data = csv.reader(f)
    data_list = list(data)
    data_list = shuffle(data_list)

    train, test = tts(data_list)



with open('train_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(train)

with open('test_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(test)

