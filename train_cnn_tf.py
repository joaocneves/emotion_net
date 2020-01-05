
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import datasets
from random import shuffle
import cv2
#import models as md
from tensorflow.python.keras.utils import to_categorical
import math
import datetime
from tensorflow.contrib.slim.python.slim.nets.alexnet import alexnet_v2


def loss_gtsrb(logits, targets):
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return cross_entropy_mean


def train_step(loss_value, gn):

    #gn = tf.Print(gn, ["GN: ", gn])

    model_learning_rate = tf.train.exponential_decay(learning_rate, gn, decay_epochs, lr_decay,
                                                     staircase=True)
    #model_learning_rate = tf.Print(model_learning_rate, ["LR: ", model_learning_rate])

    # Create optimizer
    my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
    # Initialize train step
    train_step = my_optimizer.minimize(loss_value)

    return train_step


def accuracy_of_batch(logits, targets):
    targets = tf.squeeze(tf.cast(targets, tf.int32))

    #logits = tf.Print(logits, ["LOGITS: ", logits, tf.shape(logits)], summarize=50)

    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

    #targets = tf.Print(targets, ["TRUE: ", targets, tf.shape(targets)], summarize=50)
    #batch_predictions = tf.Print(batch_predictions, ["PRED: ", batch_predictions, tf.shape(batch_predictions)], summarize=50)
    predicted_correctly = tf.equal(batch_predictions, targets)
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy



# ##########################
# Configs
IMG_EXTENSION = '.jpg'
LEARNING_FOLDER = os.path.join('GTSRB_Recognition', 'Training')
TEST_FOLDER = os.path.join('GTSRB_Recognition', 'Test', 'Images')
DEBUG_FOLDER = os.path.join('GTSRB_Recognition', 'Debug')


if not os.path.isdir(DEBUG_FOLDER):
    os.mkdir(DEBUG_FOLDER)

now = datetime.datetime.now()
name_dir = now.strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(os.path.join(DEBUG_FOLDER, name_dir))

IMDB_FILE = 'GTSRB.dat'


TOT_EPOCHS = 50

BALANCED_DATASET = 0
if len(sys.argv) > 1:
    BALANCED_DATASET = int(sys.argv[1])

DATA_AUGMENTATION = 0
if len(sys.argv) > 2:
    DATA_AUGMENTATION = int(sys.argv[2])

DROPOUT_PROB = 0.7
if len(sys.argv) > 3:
    DROPOUT_PROB = float(sys.argv[3])


LEARNING_BATCH_SIZE = 1
VALIDATION_BATCH_SIZE = 100  # type: int
PROPORTION_VALIDATION = 0.15

learning_rate = 0.01
lr_decay = 0.9
decay_epochs = 10

SHOW_WORST_CASES = []
HEIGHT_IMGS = 224
WIDTH_IMGS = 224
DEPTH_IMGS = 1
TOT_CLASSES = 10

########################################################################################################################
# TENSORFLOW
########################################################################################################################

###############################
# Create Placeholders

x_input_shape = (None, HEIGHT_IMGS, WIDTH_IMGS, DEPTH_IMGS)
x_inputs = tf.placeholder(tf.float32, shape=x_input_shape)
y_targets = tf.placeholder(tf.int32, shape=None)
y_model = tf.placeholder(tf.float32, shape=(None, TOT_CLASSES))
dropout_prob = tf.placeholder(tf.float32)
generation_num = tf.Variable(0, trainable=False)


###############################
# Load Data

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images[:1000,:,:]
train_labels = train_labels[:1000]
train_images_res = np.zeros((train_images.shape[0],224,224))
for i in range(train_images.shape[0]):
    train_images_res[i,:,:] = cv2.resize(train_images[i,:,:], (224,224))
train_images_res = np.expand_dims(train_images_res,axis=3)

train_emotions = np.load('emotions_labels.npy')
train_images = train_images_res
train_labels = to_categorical(train_labels, num_classes=10).astype('int32')

train_images, test_images = train_images / 255.0, test_images / 255.0

##

input_1 = tf.keras.Input(shape=(28,28,1))
x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(input_1)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2,2)(x)
x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
out_feat = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.Dense(64, activation='relu')(out_feat)
output_class = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=[input_1], outputs=[output_class])


with tf.variable_scope('model_definition') as scope:
    model_outputs = alexnet_v2(x_inputs, dropout_keep_prob=dropout_prob, num_classes=TOT_CLASSES)
    scope.reuse_variables()

# with tf.variable_scope('model_definition') as scope:
#     model_outputs = output_class
#     scope.reuse_variables()

loss = loss_gtsrb(model_outputs, y_targets)
accuracy = accuracy_of_batch(y_model, y_targets)
train_op = train_step(loss, generation_num)

predictions_batch = tf.cast(tf.argmax(model_outputs, 1), tf.int32)

sess = tf.Session()
plt.ion()
saver = tf.train.Saver()

########################################################################################################################
# LEARNING MAIN ()
########################################################################################################################


init = tf.global_variables_initializer()
sess.run(init)

shuffle(train_images)
shuffle(test_images)

learning_samples = []
validation_samples = []
for idx, y in enumerate(train_images):
    if idx < len(train_images)*PROPORTION_VALIDATION:
        validation_samples.append(y)
    else:
        learning_samples.append(y)




train_loss = []
train_accuracy = []
validation_accuracy = []

for e in range(TOT_EPOCHS):
    i = 0
    sess.run(generation_num.assign(e))
    epoch_loss = 0
    epoch_acc = 0
    while i < len(learning_samples):
        rand_idx = range(i, np.min([i+LEARNING_BATCH_SIZE, len(learning_samples)]))
        rand_imgs = train_images[[i],:]
        rand_y = train_labels[[i],:]
        sess.run(train_op, feed_dict={x_inputs: rand_imgs, y_targets: rand_y, dropout_prob: DROPOUT_PROB})
        [t_loss, y_out] = sess.run([loss, model_outputs], feed_dict={x_inputs: rand_imgs, y_targets: rand_y,
                                                                     dropout_prob: DROPOUT_PROB})
        t_acc = sess.run(accuracy, feed_dict={y_model: y_out, y_targets: rand_y})
        print('Learning-Epoch\t\t{}/{}\tBatch {}/{}\tLoss={:.5f}\tAcc={:.2f}'.format(e+1,
            TOT_EPOCHS, (i + 1)//LEARNING_BATCH_SIZE+1, math.ceil(len(learning_samples) / LEARNING_BATCH_SIZE),
                                                                                     t_loss, t_acc*100))
        i += LEARNING_BATCH_SIZE
        epoch_loss += t_loss*len(rand_idx)
        epoch_acc += t_acc * len(rand_idx)

    epoch_loss /= len(learning_samples)
    epoch_acc /= len(learning_samples)
    train_loss.append(epoch_loss)
    train_accuracy.append(epoch_acc)

    i = 0
    epoch_acc = 0
    while i < len(validation_samples):
        rand_idx = range(i, np.min([i + VALIDATION_BATCH_SIZE, len(validation_samples)]))
        rand_imgs, rand_y = read_batch(validation_samples, rand_idx, imdb, DATA_AUGMENTATION)

        temp_validation_y = sess.run(model_outputs, feed_dict={x_inputs: rand_imgs,
                                                               dropout_prob: DROPOUT_PROB})
        t_acc = sess.run(accuracy, feed_dict={y_model: temp_validation_y, y_targets: rand_y})
        print('Validation-Epoch\t{}/{}\tBatch\t{}/{}\tAcc={:.2f}'.format(e+1, TOT_EPOCHS,
            (i + 1)//VALIDATION_BATCH_SIZE+1, math.ceil(len(validation_samples) / VALIDATION_BATCH_SIZE), t_acc*100))
        i += VALIDATION_BATCH_SIZE
        epoch_acc += t_acc * len(rand_idx)

    epoch_acc /= len(validation_samples)
    validation_accuracy.append(epoch_acc)

    if epoch_acc == max(validation_accuracy):
        saver.save(sess, './best_model')

    eval_indices = range(1, e+2)
    plt.clf()
    plt.subplot(211)
    plt.plot(eval_indices, train_loss, 'ko-', label='Loss')
    plt.legend(loc='upper right')
    plt.title('Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.grid(which='major', axis='both')

    plt.subplot(212)
    plt.plot(eval_indices, train_accuracy, 'ko-', label='Train Set Accuracy')
    plt.plot(eval_indices, validation_accuracy, 'ro--', label='Validation Set Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(which='both', axis='y')

    plt.show()
    plt.pause(0.01)