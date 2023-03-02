# want to hardcode some rules to classify each data point, then guess the weights for each data point
from rulesBased.classifyMAC import classifyMAC
import numpy as np
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt

# defaults to current system time
random.seed()

ACCURACY_TARGET     = 0.99
NUM_EPOCHS          = 10

# precision of the visualization of the model io
VISUAL_PREC         = 100

# training points will be 9 times test points
TEST_POINTS         = 10_000

MAX_CLASSIFICATION  = 3
VALID               = 1
INVALID             = 0

def gen_classifcation():
    return random.randint(0, MAX_CLASSIFICATION)


def gen_test_point():
    data = np.array([gen_classifcation(), gen_classifcation()])
    label = sum(data)
    return data, label

def gen_points(num_points):
    data = np.ndarray((num_points, 2))
    labels = np.ndarray((num_points, 1))
    
    for i in range(num_points):
        data[i], labels[i] = gen_test_point()

    # normalize data to be between 0 and 1
    data /= MAX_CLASSIFICATION

    return data, labels

# defining the training data
train_data, train_labels    = gen_points(TEST_POINTS * 9)
test_data,  test_labels     = gen_points(TEST_POINTS)

def visualizeModel(model):
    xs = np.ndarray((VISUAL_PREC * VISUAL_PREC, 2))
    for i in range(VISUAL_PREC * VISUAL_PREC):
        xs[i][0] = i // VISUAL_PREC
        xs[i][1] = i %  VISUAL_PREC
    xs /= VISUAL_PREC
    ys = model.predict(xs)
    

    plt.scatter(xs[:,0], xs[:,1], c=np.argmax(ys, axis=1))
    plt.xlabel('Source MAC')
    plt.ylabel('Destination MAC')
    plt.colorbar()
    plt.show()
    
   

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    # Check accuracy of test data
    if logs.get('accuracy') > ACCURACY_TARGET:
        print('\nStopping training due to > {} accuracy!'.format(ACCURACY_TARGET))
        self.model.stop_training = True

callbacks = myCallback()

# defining the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64,                       activation=tf.nn.relu),
    tf.keras.layers.Dense(64,                       activation=tf.nn.relu),
    tf.keras.layers.Dense(2 * MAX_CLASSIFICATION + 1,   activation=tf.nn.softmax)
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(), 
    loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
    metrics = ['accuracy']
)

os.system('clear')

# training the model
model.fit(train_data, train_labels, epochs = NUM_EPOCHS, callbacks=[callbacks])

# evaluating test data
model.evaluate(test_data, test_labels)

model.summary()

visualizeModel(model)