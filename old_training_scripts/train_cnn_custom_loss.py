
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
#from create_initial_emotions_mnist import create_initial_emotions, generate_colors

class NewCallback(keras.callbacks.Callback):
    """ NewCallback descends from Callback
    """
    def __init__(self, data, labels, emotions):
        """ Save params in constructor
        """
        self.data = data
        self.labels = labels
        self.emotions = emotions

    def on_epoch_end(self, epoch, logs={}):
        layer_outputs = self.model.layers[-5].output
        # Extracts the outputs of the top 12 layers
        activation_model = models.Model(inputs=self.model.input,
                                        outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input



        activation_feat = activation_model.predict(self.data)
        sample_labels = self.labels
        emotion_idx, emotion_clusters, samples_feat_tsne = create_initial_emotions(activation_feat, sample_labels, 1)
        emotion_idx = self.emotions

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        colors_bank = generate_colors(10)
        # colors_bank = np.random.rand(10,3)
        plt.figure(1)
        label = []
        for i in range(10):
            plt.scatter(samples_feat_tsne[sample_labels == i, 0], samples_feat_tsne[sample_labels == i, 1], s=10,
                        c=colors_bank[i, :], alpha=1)
            label.append(mpatches.Patch(color=colors_bank[i, :], label=str(i)))

        plt.legend(handles=label)
        plt.show(block=False)

        plt.figure(2)
        label = []
        for i in range(3):
            plt.scatter(samples_feat_tsne[emotion_idx == i, 0], samples_feat_tsne[emotion_idx == i, 1], s=10,
                        c=colors_bank[i, :], alpha=1)
            label.append(mpatches.Patch(color=colors_bank[i, :], label=str(i)))
            plt.show()

        plt.legend(handles=label)
        plt.show()
        print('o')

class CustomMetrics(keras.callbacks.Callback):

    def on_batch_end(self, epoch, logs=None):

        layer_outputs = [layer.output for layer in self.model.layers[-3]]
        # Extracts the outputs of the top 12 layers
        activation_model = models.Model(inputs=self.model.input,
                                        outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input
        activations = activation_model.predict(img_tensor)
        w = self.model.layers[0].output
        #fig1 = plt.figure(1)
        #plt.cla()
        print(w)
        plt.bar(range(w[0].shape[1]), w[0][314])
        plt.gca().set_ylim(-0.15,0.15)
        im = (w[0] - w[0].min()) * 255 / (w[0].max() - w[0].min())
        #plt.imshow(im.astype(np.uint8))
        plt.draw()
        plt.pause(0.001)

        w = self.model.layers[2].get_weights()
        print(w)
        fig2 = plt.figure(2)
        plt.cla()
        im = (w[0] - w[0].min()) * 255 / (w[0].max() - w[0].min())
        plt.imshow(im.astype(np.uint8))
        plt.draw()
        plt.pause(0.001)

def penalized_loss(y_true, y_pred):
        print(y_true.shape)
        print(y_true)
        print(y_pred.shape)
        print(y_pred)
        print('---')
        d = y_pred
        return tf.reduce_mean(tf.square(d))



print(tf.__version__)
USE_EMOTIONS = True

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_emotions = np.load('emotions_labels.npy')

train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels, num_classes=10)

input_1 = Input(shape=(28,28,1))
x = layers.Conv2D(32, (3,3), activation='relu')(input_1)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
out_feat = layers.Flatten()(x)

x = layers.Dense(64, activation='relu')(out_feat)
output_class = layers.Dense(10, activation='softmax')(x)

if USE_EMOTIONS:
    x = layers.Dense(32, activation='relu')(out_feat)
    output_emotion = layers.Dense(3, activation='softmax')(x)
    model = Model(inputs=[input_1], outputs=[output_class, output_emotion])
else:
    model = Model(inputs=[input_1], outputs=[output_class])



model.summary()

model.compile(optimizer='adam',
              loss=penalized_loss,
              metrics=['accuracy'],
              run_eagerly=True)

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

if USE_EMOTIONS:
    history = model.fit(train_images, [train_labels, train_emotions], batch_size=2, epochs=10, validation_data=(test_images,  [test_labels, np.zeros_like(test_labels)]))
    np.savetxt('cnn.txt',
               np.column_stack([history.history['dense_2_accuracy'], history.history['val_dense_2_accuracy']]))
else:
    history = model.fit(train_images, train_labels, validation_data=(test_images,  test_labels), batch_size=2, epochs=10)
    np.savetxt('cnn_custom_loss.txt',
               np.column_stack([history.history['accuracy'], history.history['val_accuracy']]))


# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# print(test_acc)