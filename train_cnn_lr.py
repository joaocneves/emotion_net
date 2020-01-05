from keras.models import  Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras import backend as k
from keras import losses
from keras import datasets
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from math import ceil, floor
from sklearn.metrics import mean_squared_error
from math import sqrt

# model = Sequential()
# model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
# model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
# model.add(Dense(8, kernel_initializer='uniform', activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# inputs = np.random.random((1, 8))
# outputs = model.predict(inputs)
# targets = np.random.random((1, 8))
# rmse = sqrt(mean_squared_error(targets, outputs))




# print("===BEFORE WALKING DOWN GRADIENT===")
# print("outputs:\n", outputs)
# print("targets:\n", targets)
# print("RMSE:", rmse)

def accuracy_of_batch(logits, targets):
    targets = tf.squeeze(tf.cast(targets, tf.int32))

    logits = tf.Print(logits, ["LOGITS: ", logits, tf.shape(logits)], summarize=50)

    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

    #targets = tf.Print(targets, ["TRUE: ", targets, tf.shape(targets)], summarize=50)
    batch_predictions = tf.Print(batch_predictions, ["PRED: ", batch_predictions, tf.shape(batch_predictions)], summarize=50)
    predicted_correctly = tf.equal(batch_predictions, targets)
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy

def myfit(model=None,
          x=None,
          y=None,
          batch_size=None,
          learning_rate=None,
          learning_decay=None,
          epochs=1):

    n_samples = x.shape[0]
    n_batches = ceil(n_samples/batch_size)

    for ep in range(epochs):



        for b in range(n_batches):

            b_start = b*batch_size
            b_end = min((b+1)*batch_size, n_samples)
            x_b = x[b_start:b_end,:]
            y_b = y[b_start:b_end]

            # If your target changes, you need to update the loss
            loss = losses.sparse_categorical_crossentropy(y_b, model.output)

            #  ===== Symbolic Gradient =====
            # Tensorflow Tensor Object
            gradients = k.gradients(loss, model.trainable_weights)

            # ===== Numerical gradient =====
            # Numpy ndarray Objcet
            evaluated_gradients = sess.run(gradients, feed_dict={model.input: x_b})

            # For every trainable layer in the network
            for i in range(len(model.trainable_weights)):

                layer = model.trainable_weights[i]  # Select the layer

                # And modify it explicitly in TensorFlow
                sess.run(tf.assign_sub(layer, learning_rate * evaluated_gradients[i]))

            # decrease the learning rate
            #learning_rate *= learning_decay

            outputs = model.predict(inputs)
            centropy = sess.run(tf.reduce_mean(loss), feed_dict={model.input: x_b})
            print("crossentropy:", centropy)

        learning_rate *= learning_decay
        acc = accuracy_of_batch(model.output, y)
        acc = sess.run(acc, feed_dict={model.input: x})
        #centropy = losses.sparse_categorical_crossentropy(targets, model.output)
        #rmse = sqrt(mean_squared_error(targets, outputs))

        print("crossentropy:", centropy)
        print("acc:", acc)

if __name__ == "__main__":


    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images[:5000, :, :]
    train_labels = train_labels[:5000]
    train_images = np.expand_dims(train_images, axis=3)

    train_images, test_images = train_images / 255.0, test_images / 255.0

    targets = train_labels
    inputs = train_images

    input_1 = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(input_1)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    out_feat = Flatten()(x)

    x = Dense(64, activation='relu')(out_feat)
    output_class = Dense(10, activation='softmax')(x)

    # x = Dense(32, activation='relu')(out_feat)
    # output_emotion = Dense(3, activation='softmax')(x)

    model = Model(inputs=[input_1], outputs=[output_class])
    # model = Model(inputs=[input_1], outputs=[output_class, output_emotion])
    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Begin TensorFlow
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    myfit(model=model, x=train_images, y=train_labels, batch_size=16, learning_rate=0.01, learning_decay=0.95, epochs=50)



    loss = tf.reduce_mean(losses.sparse_categorical_crossentropy(targets, model.output))
    centropy = sess.run(loss, feed_dict={model.input: inputs})






    #final_outputs = model.predict(inputs)
    #final_rmse = sqrt(mean_squared_error(targets, final_outputs))
    centropy = sess.run(loss, feed_dict={model.input: inputs})
    print("===AFTER STEPPING DOWN GRADIENT===")
    print("outputs:\n", centropy)
    #print("targets:\n", targets)