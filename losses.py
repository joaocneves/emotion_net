import tensorflow as tf

__EPS = 1e-5


#@tf.function
def SparseCategCrossEntropy(y_true, y_pred, num_classes):

    y_pred = tf.keras.backend.clip(y_pred, __EPS, 1 - __EPS)

    one_hot_labels = tf.one_hot(y_true, num_classes)

    logits_true = tf.reduce_max(tf.multiply(one_hot_labels, y_pred), axis=1)
    loss_per_sample_batch_f = -tf.math.log(logits_true)

    # summing both loss values along batch dimension
    loss = tf.math.reduce_mean(loss_per_sample_batch_f)  # (batch_size,)

    return loss, loss_per_sample_batch_f


@tf.function
def SparseCategCrossEntropy_Emotions(y_true, y_pred, emo, num_classes):

    y_pred = tf.keras.backend.clip(y_pred, __EPS, 1 - __EPS)

    one_hot_labels = tf.one_hot(y_true, num_classes)

    logits_true = tf.reduce_max(tf.multiply(one_hot_labels, y_pred), axis=1)
    loss_per_sample_batch_f = -tf.math.log(logits_true)
    loss_emo = tf.multiply(loss_per_sample_batch_f, 1 + emo)

    # summing both loss values along batch dimension
    loss = tf.math.reduce_mean(loss_emo)  # (batch_size,)

    return loss, loss_per_sample_batch_f


@tf.function
def CenterLoss(embeddings, y_true, centers, num_classes, alpha):

    y_true_one_hot = tf.one_hot(y_true, num_classes)
    delta_centers = tf.linalg.matmul(tf.transpose(y_true_one_hot), (tf.linalg.matmul(y_true_one_hot, centers) - embeddings))  # 10x2
    center_counts = tf.math.reduce_sum(tf.transpose(y_true_one_hot), axis=1, keepdims=True) + 1  # 10x1

    delta_centers /= center_counts
    new_centers = centers - alpha * delta_centers

    loss = embeddings - tf.linalg.matmul(y_true_one_hot, new_centers)
    loss = tf.math.reduce_sum(loss ** 2, axis=1, keepdims=True)  # / K.dot(x[1], center_counts)
    loss = tf.math.reduce_mean(loss)

    return loss, new_centers

#@tf.function
def EmoLoss(y_true, y_pred, cmat, embeddings, centers, num_classes):

    y_pred = tf.keras.backend.clip(y_pred, __EPS, 1 - __EPS)

    y_true_one_hot = tf.one_hot(y_true, num_classes)
    delta_centers = tf.linalg.matmul(tf.transpose(y_true_one_hot),
                                     (tf.linalg.matmul(y_true_one_hot, centers) - embeddings))  # 10x2
    center_counts = tf.math.reduce_sum(tf.transpose(y_true_one_hot), axis=1, keepdims=True) + 1  # 10x1

    delta_centers /= center_counts
    new_centers = centers - 0.5 * delta_centers

    loss_per_sample_batch_f = tf.zeros((len(y_true),1), dtype=tf.float32)
    for i in range(len(y_true)):

        d_inter = tf.broadcast_to(embeddings[i,:], [num_classes,2]) - new_centers
        d_inter = tf.math.reduce_sum(d_inter**2, axis=1)

        cmat_norm = cmat/(tf.reduce_sum(tf.reduce_sum(cmat))+__EPS)

        loss = cmat_norm[y_true[i].numpy(),:]/tf.math.log(d_inter+1)
        loss = tf.math.reduce_sum(loss) - cmat_norm[y_true[i].numpy(),y_true[i].numpy()]/tf.math.log(d_inter[y_true[i].numpy()]+1) # discount true label

        indices = tf.constant([[y_true[i].numpy(), tf.argmax(y_pred[i,:]).numpy()]])
        updates = tf.ones([1], dtype=tf.float32)
        cmat = tf.tensor_scatter_nd_add(cmat, indices, updates)

        indices = tf.constant([[i,0]])
        updates = tf.constant([loss.numpy()])
        loss_per_sample_batch_f = tf.tensor_scatter_nd_update(loss_per_sample_batch_f, indices, updates)

    # summing both loss values along batch dimension
    loss = tf.math.reduce_mean(loss_per_sample_batch_f)  # (batch_size,)

    return loss, loss_per_sample_batch_f, cmat, new_centers
