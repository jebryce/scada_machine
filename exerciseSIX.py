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
NUM_EPOCHS          = 1

# precision of the visualization of the model io
VISUAL_PREC         = 10

# training points will be 9 times test points
TEST_POINTS         = 10_000

NUM_CRITERIA        = 4

MAX_CLASSIFICATION  = 3
VALID               = 1
INVALID             = 0


def gen_test_point():
    # generates an array of values between 0 and 1
    data = np.random.rand(NUM_CRITERIA,)
    label = data[0] + data[1] + data[2] - data[3]
    if label < 0:
        label = 0
    elif label > MAX_CLASSIFICATION:
        label = MAX_CLASSIFICATION
    
    return data, label

def gen_points(num_points):
    data = np.ndarray((num_points, NUM_CRITERIA))
    labels = np.ndarray((num_points, 1))
    
    for i in range(num_points):
        data[i], labels[i] = gen_test_point()

    return data, labels

# defining the training data
train_data, train_labels    = gen_points(TEST_POINTS * 9)
test_data,  test_labels     = gen_points(TEST_POINTS)

def visualizeModel(model):
    xs = np.ndarray((VISUAL_PREC ** NUM_CRITERIA, NUM_CRITERIA))
    for i in range(VISUAL_PREC ** NUM_CRITERIA):
        xs[i][0] = i // ( VISUAL_PREC ** 3 )
        xs[i][1] = i %  ( VISUAL_PREC ** 3 ) // ( VISUAL_PREC ** 2 )
        xs[i][2] = i %  ( VISUAL_PREC ** 2 ) // VISUAL_PREC
        xs[i][3] = i % VISUAL_PREC
    xs /= VISUAL_PREC

    ys = model.predict(xs)

    x_plot = np.ndarray((VISUAL_PREC ** NUM_CRITERIA,2))
    x_plot[:,0] = xs[:,0] + ( xs[:,1] * VISUAL_PREC )
    x_plot[:,1] = xs[:,2] + ( xs[:,3] * VISUAL_PREC )
    x_plot /= VISUAL_PREC

    plt.scatter(xs[:,0], xs[:,3], c=np.argmax(ys, axis=1))
    plt.xlabel('Average MAC')
    plt.ylabel('Average IP')
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
    tf.keras.layers.Dense(64,                   activation=tf.nn.relu),
    tf.keras.layers.Dense(64,                   activation=tf.nn.relu),
    tf.keras.layers.Dense(MAX_CLASSIFICATION,   activation=tf.nn.softmax)
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