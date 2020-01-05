import numpy as np
#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from tensorflow.keras import datasets, layers, models
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#debug_rows = 1:1000

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images[:, :, :] # to speed up tsne
train_labels = train_labels[:]
train_images_features = np.reshape(train_images,(train_images.shape[0],train_images.shape[1]*train_images.shape[2]))


train_images_features_norm = StandardScaler().fit_transform(train_images_features)
pca_model = PCA(n_components=50)
train_images_pca = pca_model.fit_transform(train_images_features_norm)
print('Explained variation: {}'.format(np.sum(pca_model.explained_variance_ratio_)))
print('Explained variation per principal component: {}'.format(pca_model.explained_variance_ratio_))

# tsne = TSNE(n_jobs=4).fit_transform(train_images_pca)
# tx, ty = tsne[:,0], tsne[:,1]
# tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
# ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
# tx = np.expand_dims(tx,1)
# ty = np.expand_dims(ty,1)
# np.savetxt('mnist_tsne_feat.txt', np.concatenate((tx,ty), axis=1))
t = np.loadtxt('mnist_tsne_feat.txt')
tx = t[:,0]
ty = t[:,1]
tx = np.expand_dims(tx,1)
ty = np.expand_dims(ty,1)

from PIL import Image
width = 4000
height = 3000
max_dim = 100
full_image = Image.new('RGB', (width, height))
for idx, x in enumerate(train_images):
    tile = Image.fromarray(np.uint8(x))
    rs = max(1, tile.width / max_dim, tile.height / max_dim)
    tile = tile.resize((int(tile.width / rs),
                        int(tile.height / rs)),
                       Image.ANTIALIAS)
    full_image.paste(tile, (int((width-max_dim) * tx[idx]),
                            int((height-max_dim) * ty[idx])))

#full_image.show()
colors_bank = np.random.rand(10,3)

plt.figure(1)
label = []
for i in range(10):
    plt.scatter(tx[train_labels==i], ty[train_labels==i], s=10, c=colors_bank[i,:], alpha=1)
    label.append(mpatches.Patch(color=colors_bank[i,:], label=str(i)))

plt.legend(handles=label)
plt.show(block=False)

train_labels = np.expand_dims(train_labels,1)
emotion_matrix = []
for i in range(len(train_labels)):

    sample_x = tx[i]
    sample_y = ty[i]

    rng_min_x = sample_x - 0.1/3
    rng_max_x = sample_x + 0.1/3
    rng_min_y = sample_y - 0.1/3
    rng_max_y = sample_y + 0.1/3

    idx = (tx > rng_min_x) & (tx < rng_max_x) & \
          (ty > rng_min_y) & (ty < rng_max_y)

    emotion_vec = np.histogram(train_labels[idx], bins=range(11))
    emotion_vec = emotion_vec[0]
    aux_label = emotion_vec[train_labels[i]]
    emotion_vec = np.delete(emotion_vec,train_labels[i])
    emotion_vec_norm = np.concatenate((aux_label, np.sort(emotion_vec)))
    emotion_matrix.append(np.expand_dims(emotion_vec_norm, axis=1))

emotion_matrix = np.concatenate(emotion_matrix,axis=1).transpose()
from sklearn.cluster import KMeans

#Build a network 20x20 with a weights format taken from the raw_data and activate Periodic Boundary Conditions.
clustering = KMeans(n_clusters=3, random_state=0).fit(emotion_matrix)
emotion_idx = clustering.labels_

plt.figure(2)
label = []
for i in range(3):
    plt.scatter(tx[emotion_idx==i], ty[emotion_idx==i], s=10, c=colors_bank[i,:], alpha=1)
    label.append(mpatches.Patch(color=colors_bank[i,:], label=str(i)))


plt.legend(handles=label)
plt.show()


print(emotion_matrix)
print('end')
