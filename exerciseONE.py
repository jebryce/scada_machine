# https://github.com/https-deeplearning-ai/tensorflow-1-public

import tensorflow as tf
import numpy as np
import random
from enum import Enum

# defaults to current system time
random.seed()

class Label(Enum):
    INVALID = 0
    VALID = 1

def gen_MAC_addr():
    return int.from_bytes(random.randbytes(4), byteorder='big')

def gen_test_point():
    data = gen_MAC_addr()

    # if the first 3 bytes are B827EB
    if data & 0xFFFFFF000000 == 0xB827EB000000:
        label = Label.VALID
    else:
        label = Label.INVALID

    return data, label

def gen_points(num_points):
    data = []
    labels = []
    for i in range(num_points):
        data_point, label = gen_test_point()
        data.append(data_point)
        labels.append(label)
    
    return data, labels



# defining the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 1, input_shape=[1])
])

model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

# defining the training data
# want to see if it can guess that the only acceptable mac addresses start with:
# B8:27:EB
# from 202481585881088 to 202481602658304

print(gen_points(50))


