import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import datasets, layers, models
from create_initial_emotions_mnist import create_initial_emotions

__EPS = 1e-5

#@tf.function
def custom_mse(y_true, y_pred):
    y_pred = K.clip(y_pred, __EPS, 1 - __EPS)

    logits_true = [y_pred[i,idx.numpy()] for i, idx in enumerate(y_true)]
    loss = -tf.math.log(logits_true)

    # summing both loss values along batch dimension
    loss = K.mean(loss)  # (batch_size,)

    return loss

def loss2(model, x, y, emotions, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return custom_mse(y_true=y, y_pred=y_)

def loss(model, x, y, emotions, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)
  return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets, emotions):
  with tf.GradientTape() as tape:
    loss_value_aux = loss(model, inputs, targets, emotions, training=True, )
    loss_value = loss2(model, inputs, targets, emotions, training=True, )
    print('---')
    print(loss_value_aux.numpy())
    print(loss_value.numpy())
    print('---')
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

BATCH_SIZE = 1
SHUFFLE_BUFFER_SIZE = 100
USE_EMOTIONS = False

epoch_loss_train = tf.keras.metrics.Mean()
epoch_loss_val = tf.keras.metrics.Mean()
epoch_acc_train = tf.keras.metrics.SparseCategoricalAccuracy()
epoch_acc_val = tf.keras.metrics.SparseCategoricalAccuracy()

print(tf.__version__)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_object(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    epoch_acc_train.update_state(y, logits)

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    epoch_acc_val.update_state(y, val_logits)

######################################
#              LOAD DATA
######################################

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images / 255
train_sample_idx = list(range(len(train_labels)))
train_emotions = np.load('emotions_labels.npy')
train_emotions = train_emotions.astype(np.uint8)
emotion_clusters = np.load('emotion_clusters.npy')
emotion_weights = [2,1.5,1]
emotion_weights = [1,1,1]
idx = np.argsort(emotion_clusters[:,0]/np.sum(emotion_clusters[:,0]))
#emotion_weights = emotion_weights[idx[::-1]]
tmp = np.zeros(len(emotion_weights))
for i in range(len(idx)):
    tmp[idx[i]] = emotion_weights[i]
emotion_weights = tmp

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
x = layers.Conv2D(32, (5,5), activation='relu', padding='same')(input_1)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (5,5), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(2,2)(x)
out_feat = layers.Flatten()(x)

x = layers.Dense(256, activation='relu')(out_feat)
output_class = layers.Dense(10, activation='softmax')(x)

if USE_EMOTIONS:
    x = layers.Dense(32, activation='relu')(out_feat)
    output_emotion = layers.Dense(3, activation='softmax')(x)
    model = models.Model(inputs=[input_1], outputs=[output_class, output_emotion])
else:
    model = models.Model(inputs=[input_1], outputs=[output_class])

#model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

######################################
#              TRAIN LOOP
######################################

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []
val_loss_results = []
val_accuracy_results = []

num_epochs = 500

for epoch in range(num_epochs):

    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Training loop - using batches of BATCH_SIZE
    for step, (x, y, z) in enumerate(train_dataset):
        # Optimize the model
        # loss_value, grads = grad(model, x, y)
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_value, grads = grad(model, x, y, emotion_weights[train_emotions[z.numpy()]])
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Track progress
        epoch_loss_train.update_state(loss_value)

        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        out = model(x, training=True)

        epoch_acc_train.update_state(y, out)

        #loss_value = train_step(x, y)

        # Log every 200 batches.
        if step % 200 == 0:
            print("%d/%d - train loss: %.4f - train accuracy: %.4f" %
                  (step, train_dataset.cardinality().numpy(), epoch_loss_train.result(), epoch_acc_train.result()))



    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)

    # Run a validation loop at the end of each epoch.
    # for x_batch_val, y_batch_val in val_dataset:
    #     val_logits = model(x_batch_val, training=False)
    #     epoch_acc_val.update_state(y_batch_val, val_logits)

    val_acc = epoch_acc_val.result()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))

    # Recalculate EMOTIONS

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
    # train_emotions, emotion_clusters, samples_feat_tsne = create_initial_emotions(intermediate_output, samples_label,
    #                                                                           epoch + 1)
    # v = np.zeros(len(train_sample_idx))
    # v[train_sample_idx] = train_emotions
    # train_emotions = v.astype('int')
    #
    # emotion_weights = [2, 1.5, 1]
    # # emotion_weights = [1,1,1]
    # idx = np.argsort(emotion_clusters[:, 0] / np.sum(emotion_clusters[:, 0]))
    # # emotion_weights = emotion_weights[idx[::-1]]
    # tmp = np.zeros(len(emotion_weights))
    # for i in range(len(idx)):
    #     tmp[idx[i]] = emotion_weights[i]
    # emotion_weights = tmp

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




