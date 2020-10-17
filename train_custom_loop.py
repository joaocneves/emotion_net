import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from create_initial_emotions_mnist import create_initial_emotions

def generate_colors(n):
    colors_bank = np.array([(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
                   (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
                   (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255),
                   (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0),
                   (255, 215, 180), (0, 0, 128), (128, 128, 128), (255, 255, 255), (0, 0, 0)])/255.0

    return colors_bank[:n,:]

def custom_loss(model, x, y, z):
  # model: tf.model.Keras
  # loss1_args: arguments to loss_1, as tuple.
  # loss2_args: arguments to loss_2, as tuple.
  with tf.GradientTape() as tape:
    y_ = model(x, training=True)
    l1_value = loss_object(y_true=y, y_pred=y_[0])
    l2_value = loss_object(y_true=z, y_pred=y_[1])
    loss_value = [l1_value, l2_value]
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_, sample_weight=[1,3])


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 100
USE_EMOTIONS = False

######################################
#              LOAD DATA
######################################

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images / 255
train_sample_idx = list(range(len(train_labels)))
train_emotions = np.load('emotions_labels.npy')
train_emotions = train_emotions.astype(np.uint8)

#train_images = np.reshape(train_images, (-1, 28*28)) / 255
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_sample_idx))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

val_dataset = test_dataset

######################################
#              MODEL
######################################

input_1 = layers.Input(shape=(28,28,1))
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_1)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
out_feat = layers.Flatten()(x)

x = layers.Dense(64, activation='relu')(out_feat)
output_class = layers.Dense(10, activation='softmax')(x)

if USE_EMOTIONS:
    x = layers.Dense(32, activation='relu')(out_feat)
    output_emotion = layers.Dense(3, activation='softmax')(x)
    model = models.Model(inputs=[input_1], outputs=[output_class, output_emotion])
else:
    model = models.Model(inputs=[input_1], outputs=[output_class])

#model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

######################################
#              TRAIN LOOP
######################################

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):

    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    epoch_loss_classif_avg = tf.keras.metrics.Mean()
    epoch_accuracy_classif = tf.keras.metrics.SparseCategoricalAccuracy()
    epoch_loss_emotion_avg = tf.keras.metrics.Mean()
    epoch_accuracy_emotion = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # Training loop - using batches of BATCH_SIZE
    for step, (x, y, z) in enumerate(train_dataset):
        # Optimize the model
        # loss_value, grads = grad(model, x, y)
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if USE_EMOTIONS:
            loss_values, grads = custom_loss(model, x, y, z)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # Track progress
            epoch_loss_classif_avg.update_state(loss_values[0])  # Add current batch loss
            epoch_loss_emotion_avg.update_state(loss_values[1])

            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            out = model(x, training=True)
            epoch_accuracy_classif.update_state(y, out[0])
            epoch_accuracy_emotion.update_state(z, out[1])

        else:
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # Track progress
            epoch_loss_classif_avg.update_state(loss_value)

            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            out = model(x, training=True)

            epoch_accuracy_classif.update_state(y, out)

        # Log every 200 batches.
        if step % 200 == 0:
            print("%d/%d - train loss: %.4f - train accuracy: %.4f" %
                  (step, train_dataset.cardinality().numpy(), epoch_loss_classif_avg.result(), epoch_accuracy_classif.result()))

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        val_acc_metric.update_state(y_batch_val, val_logits)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))




    layer_name = 'flatten'
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                           outputs=model.get_layer(layer_name).output)
    intermediate_output = []
    labels = []
    train_sample_idx = []
    for step, (x, y, z) in enumerate(train_dataset):
        intermediate_output.append(intermediate_layer_model(x))
        labels.append(y)
        train_sample_idx.append(z)

    intermediate_output = np.vstack(intermediate_output)
    samples_label = np.hstack(labels)
    train_sample_idx = np.hstack(train_sample_idx)
    emotion_id, emotion_clusters, samples_feat_tsne = create_initial_emotions(intermediate_output, samples_label, epoch+1)


    # SHOW

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Turn interactive plotting off
    plt.ioff()

    colors_bank = generate_colors(10)
    # colors_bank = np.random.rand(10,3)
    plt.figure(1)
    label = []
    for i in range(10):
        plt.scatter(samples_feat_tsne[samples_label == i, 0], samples_feat_tsne[samples_label == i, 1], s=10,
                    c=colors_bank[i, :], alpha=1)
        label.append(mpatches.Patch(color=colors_bank[i, :], label=str(i)))

    plt.legend(handles=label)
    #plt.show(block=False)
    tmpfilename = 'samples_feat_tsne_{0}.png'.format(epoch+1)
    plt.savefig(tmpfilename)
    plt.clf()

    plt.figure(2)
    label = []
    for i in range(3):
        plt.scatter(samples_feat_tsne[emotion_id == i, 0], samples_feat_tsne[emotion_id == i, 1], s=10,
                    c=colors_bank[i, :], alpha=1)
        label.append(mpatches.Patch(color=colors_bank[i, :], label=str(i)))

    plt.legend(handles=label)
    #plt.show()
    tmpfilename = 'samples_emotion_tsne_{0}.png'.format(epoch+1)
    plt.savefig(tmpfilename)
    plt.clf()


    # End epoch
    train_loss_results.append(epoch_loss_classif_avg.result())
    train_accuracy_results.append(epoch_accuracy_classif.result())
    if USE_EMOTIONS:
        train_loss_results.append(epoch_loss_emotion_avg.result())
        train_accuracy_results.append(epoch_accuracy_emotion.result())

    if USE_EMOTIONS:
        print("Epoch {:03d}: Loss (Classification): {:.3f}, Accuracy (Classification): {:.3%} Loss (Emotion): {:.3f}, Accuracy (Emotion): {:.3%}".format(epoch,
                                                                          epoch_loss_classif_avg.result(),
                                                                          epoch_accuracy_classif.result(),
                                                                          epoch_loss_emotion_avg.result(),
                                                                          epoch_accuracy_emotion.result()))
    else:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_classif_avg.result(),
                                                                    epoch_accuracy_classif.result()))





