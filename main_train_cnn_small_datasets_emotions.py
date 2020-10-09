import time
import os
import numpy as np
import tensorflow as tf
from scipy.io import savemat

from data_loading.mnist import load_mnist_image_format_3_splits_normalized
from data_loading.mnist import save_mnist_images_to_disk
from models.simple_models import CNN_32C5S1_P2_64C5S1_P2_F256_F2_out_F2
from losses import SparseCategCrossEntropy as SCCE_loss, CenterLoss
from losses import SparseCategCrossEntropy_Emotions as SCCE_EMO_loss
from aux_functions import getLossPerSample, getFeaturesPerSample, createExperimentNameFromParams

import matplotlib.patches as mpatches
from create_initial_emotions_mnist import generate_colors
import matplotlib.pyplot as plt
plt.ion()

def debugLearningProgress(model, train_dataset, loss_per_sample):

    intermediate_output = []
    labels = []
    train_sample_idx = []
    for step, (x, y, tidx) in enumerate(train_dataset):
        intermediate_output.append(model(x)[1])
        labels.append(y)
        train_sample_idx.append(tidx)

    intermediate_output = np.vstack(intermediate_output)
    samples_label = np.hstack(labels)
    train_sample_idx = np.hstack(train_sample_idx)

    # sort features to original order
    sort_idx = train_sample_idx.argsort()
    intermediate_output = intermediate_output[sort_idx, :]
    samples_label = samples_label[sort_idx]
    train_sample_idx = train_sample_idx[sort_idx]

    colors_bank = generate_colors(10)
    # colors_bank = np.random.rand(10,3)
    plt.figure(1)
    label = []
    for i in range(10):
        plt.scatter(intermediate_output[samples_label == i, 0], intermediate_output[samples_label == i, 1], s=10,
                    c=colors_bank[i, :], alpha=1)
        label.append(mpatches.Patch(color=colors_bank[i, :], label=str(i)))

    plt.legend(handles=label)

    import statistics
    min_loss = 4
    plt.scatter(intermediate_output[loss_per_sample > min_loss, 0], intermediate_output[loss_per_sample > min_loss, 1],
                c=np.max(loss_per_sample)-loss_per_sample[loss_per_sample > min_loss], cmap='Purples', marker='x')
    plt.show()
    plt.draw()
    plt.pause(0.001)

######################################
#              PARAMS
######################################

params = dict()
params['DATASET'] ='MNIST'
params['BATCH_SIZE'] = 16
params['SHUFFLE_BUFFER_SIZE'] = 100
params['NUM_EPOCHS'] = 100
params['LEARNING_RATE'] = 1e-3
params['TYPE'] = 'baseline'
params['DEBUG'] = True
experiment_name = os.path.join('experiments', createExperimentNameFromParams(params))
if not os.path.exists(experiment_name):
    os.mkdir(experiment_name)

######################################
#       TRAIN/TEST FUNCTIONS
######################################

epoch_loss_train = tf.keras.metrics.Mean()
epoch_loss_val = tf.keras.metrics.Mean()
epoch_acc_train = tf.keras.metrics.SparseCategoricalAccuracy()
epoch_acc_val = tf.keras.metrics.SparseCategoricalAccuracy()

optimizer = tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE'])

@tf.function
def train_step(x, y, centers_f, num_classes):

    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value_ssce_train, dummy1 = SCCE_loss(y, logits[0], num_classes)
        loss_value_emo_train, centers_f = CenterLoss(logits[1], y, centers_f, num_classes, 0.01)
        loss_value_train = loss_value_ssce_train + loss_value_emo_train
        #loss_per_sample[emo.numpy()] = loss_per_sample_batch_f

    # calculate gradients
    grads = tape.gradient(loss_value_train, model.trainable_weights)

    # provide the optmizer with gradients for updating weights
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Track progress
    epoch_acc_train.update_state(y, logits[0])
    epoch_loss_train.update_state(loss_value_train)

    return centers_f

@tf.function
def test_step(x, y, num_classes):
    val_logits = model(x, training=False)
    loss_value_val, dummy2 = SCCE_loss(y, val_logits[0], num_classes)
    epoch_acc_val.update_state(y, val_logits[0])
    epoch_loss_val.update_state(loss_value_val)



######################################
#              LOAD DATA
######################################

train_images, val_images, train_labels, val_labels, test_images, test_labels = \
    load_mnist_image_format_3_splits_normalized()

train_sample_idx = list(range(len(train_labels)))
val_sample_idx = list(range(len(val_labels)))

# the training dataset will have a third element for keeping track of its sample id
# it will be useful for using different policies for specific samples
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_sample_idx))
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels, val_sample_idx))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

train_dataset = train_dataset.shuffle(params['SHUFFLE_BUFFER_SIZE']).batch(params['BATCH_SIZE'])
val_dataset = val_dataset.batch(params['BATCH_SIZE'])
test_dataset = test_dataset.batch(params['BATCH_SIZE'])

# if params['DEBUG']:
#     save_mnist_images_to_disk(train_images, os.path.join(experiment_name, 'train_data'))
#     save_mnist_images_to_disk(val_images, os.path.join(experiment_name, 'val_data'))

######################################
#              MODEL
######################################

input_dim = [28,28,1]
output_dim = len(np.unique(train_labels))
num_classes = output_dim
model = CNN_32C5S1_P2_64C5S1_P2_F256_F2_out_F2(input_dim, output_dim)


######################################
#              TRAIN LOOP
######################################

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []
val_loss_results = []
val_accuracy_results = []

centers = tf.zeros([10, 2], tf.float32)

for epoch in range(params['NUM_EPOCHS']):

    data_epoch = {'intermediate_output_train': [], 'samples_label_train': [], 'loss_per_sample_train': [],
                  'intermediate_output_val': [], 'samples_label_val': [], 'loss_per_sample_val': []}

    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Training loop - using batches of BATCH_SIZE
    for step, (x, y, tidx) in enumerate(train_dataset):

        train_step(x, y, centers, num_classes)

        # Log every 200 batches.
        if step % 500 == 0:
            print("%d/%d - train loss: %.4f - train accuracy: %.4f" %
                  (step, train_dataset.cardinality().numpy(), epoch_loss_train.result(), epoch_acc_train.result()))

    if params['DEBUG']:

        data_epoch['intermediate_output_train'], data_epoch['samples_label_train'] = getFeaturesPerSample(model, train_dataset)
        # compute loss of training samples
        num_train_samples = len(train_labels)
        data_epoch['loss_per_sample_train'] = getLossPerSample(model, train_dataset, num_train_samples, num_classes)
        debugLearningProgress(model, train_dataset, data_epoch['loss_per_sample_train'])


        data_epoch['intermediate_output_val'], data_epoch['samples_label_val'] = getFeaturesPerSample(model, val_dataset)
        # compute loss of validation samples
        num_val_samples = len(val_labels)
        data_epoch['loss_per_sample_val'] = getLossPerSample(model, val_dataset, num_val_samples, num_classes)


        savemat(os.path.join(experiment_name,'debug_data_epoch_{0}.mat'.format(epoch)), data_epoch)


    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val, tmp in val_dataset:
        test_step(x_batch_val, y_batch_val, num_classes)

    val_acc = epoch_acc_val.result()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))

    # Save Results
    train_loss_results.append(epoch_loss_train.result().numpy())
    train_accuracy_results.append(epoch_acc_train.result().numpy())
    val_loss_results.append(epoch_loss_val.result().numpy())
    val_accuracy_results.append(epoch_acc_val.result().numpy())

    # Reset training metrics at the end of each epoch
    epoch_acc_train.reset_states()
    epoch_acc_val.reset_states()

    if params['DEBUG']:
        np.savetxt(os.path.join(experiment_name, 'train_loss_results.txt'), train_loss_results)
        np.savetxt(os.path.join(experiment_name, 'train_accuracy_results.txt'), train_accuracy_results)
        np.savetxt(os.path.join(experiment_name, 'val_loss_results.txt'), val_loss_results)
        np.savetxt(os.path.join(experiment_name, 'val_accuracy_results.txt'), val_accuracy_results)

