# https://github.com/https-deeplearning-ai/tensorflow-1-public


# pip install tensorflow-macos
import tensorflow as tf
import numpy as np
import random
from enum import Enum
import os
import visualizeDATA

# defaults to current system time
random.seed()

CORRECT_PREFIXES    = (0x000000_000000,
                       0xB827EB_000000,
                       0xCA7CA7_000000,
                       0xFAB004_000000)

CORRECT_MASK        = 0xFFFFFF_000000
MAX_NUMBER          = 0xFFFFFF_FFFFFF

ACCURACY_TARGET     = 0.95

# training points will be 9 times test points
TEST_POINTS         = 100_000

NUM_EPOCHS          = 100

# percent of points that will be correct
PERCENT_CORRECT = 25
if PERCENT_CORRECT > 100 or PERCENT_CORRECT < 0:
    raise ValueError("PERCENT_CORRECT must be >= 0 and <= 100")

VIS_PREC = 0xFFFF

class Label(Enum):
    INVALID = 0
    VALID = 1

def gen_MAC_addr():
    generated = random.randint(0, MAX_NUMBER)
    if random.randint(1,100) <= PERCENT_CORRECT:
        choice = random.randrange(0, len(CORRECT_PREFIXES))
        correct_generated = CORRECT_PREFIXES[choice] + (generated & ~CORRECT_MASK)
        return correct_generated
    else:
        return generated

def gen_test_point():
    data = gen_MAC_addr()

    if data & CORRECT_MASK in CORRECT_PREFIXES:
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
        data[i] =   data_point / MAX_NUMBER
        labels[i] = label
    
    return data, labels

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):

    # Check accuracy of test data
    if epoch < 10:
        return

    if logs.get('accuracy') > ACCURACY_TARGET:
        print('\nStopping training due to > {} accuracy!'.format(ACCURACY_TARGET))
        self.model.stop_training = True
    
    

# Instantiate class
callbacks = myCallback()

# defining the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024,     activation=tf.nn.relu),
    tf.keras.layers.Dense(1,        activation=tf.nn.sigmoid)
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(), 
    loss = tf.keras.losses.MeanSquaredError(), 
    metrics = ['accuracy']
)

# defining the training data
train_data, train_labels    = gen_points(TEST_POINTS * 9)
test_data,  test_labels     = gen_points(TEST_POINTS)

num_valid = 0
for i in range(len(train_labels)):
    if train_labels[i] == 1:
        num_valid += 1

os.system('clear')
print("Number of valid training points: ", num_valid)
print("Percentage of training points that are valid: {:.1f}%".format(100 * num_valid / TEST_POINTS / 9))

# training the model
model.fit(train_data, train_labels, epochs = NUM_EPOCHS, callbacks=[callbacks])

# evaluating test data
model.evaluate(test_data, test_labels)

xs = np.linspace(0, MAX_NUMBER, VIS_PREC)
ys = model.predict(xs / MAX_NUMBER)
visualizeDATA.plot(xs, ys, CORRECT_PREFIXES)