

import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from create_initial_emotions_mnist import create_initial_emotions
from create_initial_emotions_mnist import generate_colors
from sklearn.cluster import KMeans

__EPS = 1e-5
NUM_EMOTIONS = 4

def getLossPerSample(train_dataset, emo):

    loss_per_sample_f = np.zeros((len(train_labels),))

    # Loss Visualization Loop
    for step, (x, y, tidx) in enumerate(train_dataset):
        loss_per_sample_batch_f = loss_vis_step(x, y, emo[tidx.numpy()], NUM_CLASSES)
        loss_per_sample_f[tidx.numpy()] = loss_per_sample_batch_f

    train_sample_idx
    #highest_loss = loss_per_sample.argsort()[-5:]

    return loss_per_sample_f


def showMostProblematic(train_images, train_sample_idx, highest_loss):

    from mpl_toolkits.axes_grid1 import ImageGrid

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for i in range(5):
        ax = grid[i]
        im = train_images[train_sample_idx[highest_loss[i]], :]
        ax.imshow(im)
    plt.show()

def debugLearningProgress(model, train_dataset, loss_per_sample):

    layer_name = 'flatten'
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                              outputs=model.get_layer(layer_name).output)

    intermediate_output = []
    labels = []
    train_sample_idx = []
    for step, (x, y, tidx) in enumerate(train_dataset):
        intermediate_output.append(intermediate_layer_model(x))
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

    train_emotions, emotion_clusters, samples_feat_tsne = create_initial_emotions(intermediate_output, samples_label,
                                                                                  epoch + 1)

    colors_bank = generate_colors(10)
    # colors_bank = np.random.rand(10,3)
    plt.figure(1)
    label = []
    for i in range(10):
        plt.scatter(samples_feat_tsne[samples_label == i, 0], samples_feat_tsne[samples_label == i, 1], s=10,
                    c=colors_bank[i, :], alpha=1)
        label.append(mpatches.Patch(color=colors_bank[i, :], label=str(i)))

    plt.legend(handles=label)

    import statistics
    min_loss = 4
    plt.scatter(samples_feat_tsne[loss_per_sample > min_loss, 0], samples_feat_tsne[loss_per_sample > min_loss, 1],
                c=np.max(loss_per_sample)-loss_per_sample[loss_per_sample > min_loss], cmap='Purples', marker='x')
    plt.show()


#@tf.function
def CategCrossEntropy(y_true, y_pred, emo, num_classes):

    y_pred = tf.keras.backend.clip(y_pred, __EPS, 1 - __EPS)

    one_hot_labels = tf.one_hot(y_true, num_classes)

    logits_true = tf.reduce_max(tf.multiply(one_hot_labels, y_pred), axis=1)
    loss_per_sample_batch_f = -tf.math.log(logits_true)
    loss_emo = tf.multiply(loss_per_sample_batch_f, 1 + emo)
    # summing both loss values along batch dimension
    loss = tf.math.reduce_mean(loss_emo)  # (batch_size,)

    return loss, loss_per_sample_batch_f


BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 100
EPOCHS = 50
NUM_CLASSES = 10

######################################
#         SPEED UP FUNCTIONS
######################################

epoch_loss_train = tf.keras.metrics.Mean()
epoch_loss_val = tf.keras.metrics.Mean()
epoch_acc_train = tf.keras.metrics.SparseCategoricalAccuracy()
epoch_acc_val = tf.keras.metrics.SparseCategoricalAccuracy()

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
loss_object = CategCrossEntropy
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


def loss_vis_step(x, y, emo, num_classes=10):

    logits = model(x, training=True)
    loss_value, loss_per_sample_batch_f = loss_object(y, logits, emo, num_classes)

    return loss_per_sample_batch_f


#@tf.function
def train_step(x, y, emo,  num_classes=10):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_object(y, logits, emo, num_classes)
        #loss_per_sample[emo.numpy()] = loss_per_sample_batch_f
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    epoch_acc_train.update_state(y, logits)
    return loss_value

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    epoch_acc_val.update_state(y, val_logits)

######################################
#              LOAD DATA
######################################

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images / 255
test_images = test_images / 255
#train_images = test_images
#train_labels = test_labels
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=1)
train_sample_idx = list(range(len(train_labels)))
train_emotions = np.zeros(len(train_sample_idx))
emotion_weights = [2,1.5,1]

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_sample_idx))
val_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


######################################
#              MODEL
######################################

input_1 = layers.Input(shape=(28,28,1))
x = layers.Conv2D(32, (5,5), activation='relu', padding='same')(input_1)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (5,5), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(2,2)(x)
out_feat = layers.Flatten()(x)

x = layers.Dense(256, activation='relu')(out_feat)
output_class = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs=[input_1], outputs=[output_class])

#model.summary()
#tf.keras.utils.plot_model(model, show_shapes=True)


######################################
#              TRAIN LOOP
######################################

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []
val_loss_results = []
val_accuracy_results = []
emotion_idx = np.zeros(len(train_labels))#*(NUM_EMOTIONS-1)

for epoch in range(EPOCHS):

    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Training loop - using batches of BATCH_SIZE
    for step, (x, y, tidx) in enumerate(train_dataset):

        loss_value, tmp = train_step(x, y, emotion_idx[tidx.numpy()], NUM_CLASSES)
        # Track progress
        epoch_loss_train.update_state(loss_value)

        # Log every 200 batches.
        if step % 200 == 0:
            print("%d/%d - train loss: %.4f - train accuracy: %.4f" %
                  (step, train_dataset.cardinality().numpy(), epoch_loss_train.result(), epoch_acc_train.result()))

    emotion_idx = np.ones(len(train_labels)) * (NUM_EMOTIONS - 1)
    loss_per_sample = getLossPerSample(train_dataset, emotion_idx)
    np.savetxt('loss_per_sample.txt', loss_per_sample)
    #plt.bar(range(len(loss_per_sample)), loss_per_sample)
    #plt.show()

    highest_loss = loss_per_sample.argsort()[-5:]
    showMostProblematic(train_images, train_sample_idx, highest_loss)
    debugLearningProgress(model, train_dataset, loss_per_sample)

    """ Find K emotions using clustering """
    import kmeans1d
    emotion_idx, emotion_clusters = kmeans1d.cluster(loss_per_sample, k=NUM_EMOTIONS)
    emotion_idx = np.array(emotion_idx)



    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)

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

    np.savetxt('train_loss_results.txt',train_loss_results)
    np.savetxt('train_accuracy_results.txt', train_accuracy_results)
    np.savetxt('val_loss_results.txt', val_loss_results)
    np.savetxt('val_accuracy_results.txt', val_accuracy_results)
