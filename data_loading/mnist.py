import os
import cv2
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split

def load_mnist_image_format_3_splits_normalized():


    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images / 255
    test_images = test_images / 255

    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2,
                                                                          random_state=1)

    return train_images, val_images, train_labels, val_labels, test_images, test_labels

def save_mnist_images_to_disk(images, out_path):

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for idx, im in enumerate(images):

        im = im*255.0
        im = im.astype('uint8')
        cv2.imwrite('%s\\%05d.jpg' % (out_path,idx), im)
