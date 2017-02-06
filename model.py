import csv
import tensorflow as tf
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from process_line import process_line
import json
from keras.models import model_from_json
import os
import argparse
from keras.regularizers import l2, activity_l2
tf.python.control_flow_ops = tf


batch_size = 200
training_samples = 12000
val_samples = 2400
epochs = 10


def gen(file):

    while 1:
        f = open(file)
        data = csv.reader(f)
        data_list = list(data)
        data_list = shuffle(data_list)

        x, y = process_line(data_list, batch_size)

        yield x, y
        f.close()


##########################################################################################################
# MODEL

def load_model(time_len=1):
    parser = argparse.ArgumentParser(description='Behavioral Cloning')
    parser.add_argument('model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')

    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.loads(jfile.read()))

    weights_file = args.model.replace('json', 'h5')

    if os.path.exists(weights_file):
        model.load_weights(weights_file)

    model.compile(optimizer="adam", loss="mse", lr=.0001)
    return model


def model(time_len=1):

    model = Sequential()

    row, col, ch = 66, 200, 3  # (height, width, channels)
    # model.add(Lambda(lambda x: x/255 - 0.5,
    #                  input_shape=(row, col, ch),
    #                  output_shape=(row, col, ch)))
    model.add(Convolution2D(3, 5, 5, subsample=(2, 2), border_mode="same", input_shape=(row, col, ch)))
    model.add(ELU())
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(48, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    # model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


model = model()
# model = load_model()

model.fit_generator(
    gen('train_data.csv'),
    samples_per_epoch=training_samples,
    nb_epoch=epochs,
    validation_data=gen('test_data.csv'),
    nb_val_samples=val_samples
  )

print("Saving model weights and configuration file.")


model.save_weights("model.h5")
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)