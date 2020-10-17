import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import datasets, layers


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

# Setting seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Dataset: given 2 numbers, predict the sum
# Sum 2 numbers from 0 to 10 dataset
samples = train_images
targets = train_labels

# Samples for testing
samples_test = test_images
targets_test = test_labels

# Model
input_1 = Input(shape=(28,28,1),name='img_input')
xl = layers.Conv2D(32, (3,3), activation='relu')(input_1)
xl = layers.MaxPooling2D((2,2))(xl)
xl = layers.Conv2D(64, (3,3), activation='relu')(xl)
xl = layers.MaxPooling2D(2,2)(xl)
xl = layers.Conv2D(64, (3,3), activation='relu')(xl)
out_feat = layers.Flatten()(xl)
xl = layers.Dense(64, activation='relu')(out_feat)
output_class = layers.Dense(10, activation='softmax')(xl)
model = Model(inputs=[input_1], outputs=[output_class])



# Loss
def loss_fn(y_true, y_pred):
    # You can get all the crazy and twisted you
    # want here no Keras restrictions this time :)
    loss_value = K.sum(K.pow((y_true - y_pred), 2))
    return loss_value


# Optimizer to run the gradients
optimizer = Adam(lr=1e-4)

# Graph creation
# Creating training flow
# Ground truth input, samples or X_t
y_true = Input(shape=[10])

# Prediction
y_pred = model(input_1)

# Loss
loss = loss_fn(y_true, y_pred)

# Operation for getting
# gradients and updating weights
updates_op = optimizer.get_updates(
    params=model.trainable_weights,
    loss=loss)

# The graph is created, now we need to call it
# this would be similar to tf session.run()
train = K.function(
    inputs=[input_1, y_true],
    outputs=[loss],
    updates=updates_op)

test = K.function(
    inputs=[input_1, y_true],
    outputs=[loss])

# Training loop
epochs = 100

for epoch in range(epochs):
    print('Epoch %s:' % epoch)

    # Fancy progress bar
    pbar = tqdm(range(len(samples)))

    # Storing losses for computing mean
    losses_train = []

    # Batch loop: batch size=1
    for idx in pbar:
        sample = samples[idx:idx+10]
        target = targets[idx:idx+10]

        # Adding batch dim since batch=1
        #sample = np.expand_dims(sample, axis=0)
        #target = np.expand_dims(target, axis=0)

        # To tensors, input of
        # K.function must be tensors
        sample = K.constant(sample, name='img_input')
        target = K.constant(target)

        # Running the train graph
        loss_train = train([sample, target])

        # Compute loss mean
        losses_train.append(loss_train[0])
        loss_train_mean = np.mean(losses_train)

        # Update progress bar
        pbar.set_description('train Loss: %.3f' % loss_train_mean)

    # Testing
    losses_test = []
    for idx in range(len(samples_test)):
        sample_test = samples_test[idx:idx+10]
        target_test = targets_test[idx:idx+10]

        # Adding batch dim since batch=1
        sample_test = np.expand_dims(sample_test, axis=0)
        target_test = np.expand_dims(target_test, axis=0)

        # To tensors
        sample_test = K.constant(sample_test)
        target_test = K.constant(target_test)

        # Evaluation test graph
        loss_test = test([sample_test, target_test])

        # Compute test loss mean
        losses_test.append(loss_test[0])

    loss_test_mean = np.mean(losses_test)
    print('Test Loss: %.3f' % loss_test_mean)