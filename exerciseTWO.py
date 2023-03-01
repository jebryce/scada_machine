# https://github.com/https-deeplearning-ai/tensorflow-1-public


# pip install tensorflow-macos
import tensorflow as tf
import numpy as np
import random
from enum import Enum
import os

# defaults to current system time
random.seed()

CORRECT_NUMBER = 0xB8

PERCENT_CORRECT = 25
if PERCENT_CORRECT > 100 or PERCENT_CORRECT < 0:
    raise ValueError("PERCENT_CORRECT must be >= 0 and <= 100")

class Label(Enum):
    INVALID = 0
    VALID = 1

def gen_MAC_addr():
    if random.randint(1,100) <= PERCENT_CORRECT:
        return CORRECT_NUMBER
    else:
        return random.randint(0,255)

def gen_test_point():
    data = gen_MAC_addr()

    if data == CORRECT_NUMBER:
        label = Label.VALID.value
    else:
        label = Label.INVALID.value

    return data, label

def gen_points(num_points):
    data = np.ndarray((num_points, 1))
    labels = np.ndarray((num_points, 1))
    for i in range(num_points):
        data_point, label = gen_test_point()

        # normalize data to values between 0 and 1
        data[i] =   data_point / 0xFF
        labels[i] = label
    
    return data, labels

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    # Check accuracy of test data
    prediction = model.predict(np.array([CORRECT_NUMBER]) / 0xFF)
    print("The currect prediction for [0x{:02X}] is {}".format(CORRECT_NUMBER, prediction[0]))
    if prediction > 0.95:
      print('Stopping training due to high enough model accuracy!')
      self.model.stop_training = True

# Instantiate class
callbacks = myCallback()

# defining the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(), 
    loss = tf.keras.losses.MeanSquaredError(), 
    metrics = ['accuracy']
)

# defining the training data
train_data, train_labels    = gen_points(90_000)
test_data,  test_labels     = gen_points(10_000)

num_valid = 0
for i in range(len(train_labels)):
    if train_labels[i] == 1:
        num_valid += 1

os.system('clear')
print("Number of valid points: ", num_valid)

# training the model
model.fit(train_data, train_labels, epochs = 100, callbacks=[callbacks])

# evaluating test data
model.evaluate(test_data, test_labels)

valid_nums = np.array([
    CORRECT_NUMBER
])


# Make a prediction
prediction = model.predict(valid_nums / 0xFF)
print(prediction[0])
if prediction[0] > 0.9:
    print('The value {:02X} is valid!'.format(valid_nums[0]))
elif prediction[0] < 0.1:
    print('The value {:02X} is invalid!'.format(valid_nums[0]))
else:
    print('The value {:02X} isn\'t decisively valid or invalid!'.format(valid_nums[0]))
