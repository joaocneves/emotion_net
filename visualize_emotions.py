import numpy as np
import os
from MulticoreTSNE import MulticoreTSNE as TSNE
#from sklearn.manifold import TSNE
from tensorflow.keras import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import random
import matplotlib.pyplot as plt


def generate_colors(n):
    colors_bank = np.array([(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
                   (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
                   (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255),
                   (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0),
                   (255, 215, 180), (0, 0, 128), (128, 128, 128), (255, 255, 255), (0, 0, 0)])/255.0

    return colors_bank[:n,:]

def visualize_emotions(samples_label, epoch):

    """ Asserts """
    if len(samples_label.shape) == 1:
        samples_label = np.expand_dims(samples_label, axis=1)


    """ Learn Data Correlations (TSNE) """
    tmpfilename = 'samples_feat_tsne_{0}.txt'.format(epoch)
    if os.path.isfile(tmpfilename):
        samples_feat_tsne = np.loadtxt(tmpfilename)

    """ Estimate Density in the Manifold Space """
    tx = samples_feat_tsne[:, [0]]
    ty = samples_feat_tsne[:, [1]]
    emotion_matrix = []
    for i in range(len(samples_label)):
        sample_x = tx[i]
        sample_y = ty[i]

        rng_min_x = sample_x - 0.1 / 3
        rng_max_x = sample_x + 0.1 / 3
        rng_min_y = sample_y - 0.1 / 3
        rng_max_y = sample_y + 0.1 / 3

        idx = (tx > rng_min_x) & (tx < rng_max_x) & \
              (ty > rng_min_y) & (ty < rng_max_y)

        emotion_vec = np.histogram(samples_label[idx], bins=range(11))[0]

        cur_label_nn = emotion_vec[samples_label[i]]
        emotion_vec = np.delete(emotion_vec, samples_label[i])
        emotion_vec_norm = np.concatenate((cur_label_nn, np.sort(emotion_vec)))
        emotion_matrix.append(np.expand_dims(emotion_vec_norm, axis=1))

    emotion_matrix = np.concatenate(emotion_matrix, axis=1).transpose()

    """ Find K emotions using clustering """
    clustering = KMeans(n_clusters=3, random_state=0).fit(emotion_matrix)
    emotion_idx = clustering.labels_
    emotion_clusters = clustering.cluster_centers_

    return emotion_idx, emotion_clusters, samples_feat_tsne

if __name__ == "__main__":

    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    #train_images = train_images[0:1000, :, :]
    #train_labels = train_labels[0:1000]
    # image to column vector
    train_images_features = np.reshape(train_images,(train_images.shape[0],train_images.shape[1]*train_images.shape[2]))

    samples_feat = train_images_features
    sample_labels = train_labels
    emotion_idx, emotion_clusters, samples_feat_tsne = visualize_emotions(sample_labels, 1)


    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    colors_bank = generate_colors(10)
    #colors_bank = np.random.rand(10,3)
    plt.figure(1)
    label = []
    for i in range(10):
        plt.scatter(samples_feat_tsne[sample_labels==i,0], samples_feat_tsne[sample_labels==i,1], s=10, c=colors_bank[i,:], alpha=1)
        label.append(mpatches.Patch(color=colors_bank[i,:], label=str(i)))

    plt.legend(handles=label)
    plt.show(block=False)



    plt.figure(2)
    label = []
    for i in range(3):
        plt.scatter(samples_feat_tsne[emotion_idx==i,0], samples_feat_tsne[emotion_idx==i,1], s=10, c=colors_bank[i,:], alpha=1)
        label.append(mpatches.Patch(color=colors_bank[i,:], label=str(i)))


    plt.legend(handles=label)
    plt.show()
