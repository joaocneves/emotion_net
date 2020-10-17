import numpy as np
import tensorflow as tf
from losses import SparseCategCrossEntropy as scce_loss


def getFeaturesPerSample(model, dataset):

    layer_name = 'flatten'
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                              outputs=model.get_layer(layer_name).output)

    intermediate_output = []
    labels = []
    train_sample_idx = []
    for step, (x, y, tidx) in enumerate(dataset):
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

    return intermediate_output, samples_label

def getLossPerSample(model, train_dataset, n_samples, num_classes):

    loss_per_sample_f = np.zeros((n_samples,))

    # Loss Visualization Loop
    for step, (x, y, tidx) in enumerate(train_dataset):

        logits = model(x, training=True)
        tmp, loss_per_sample_batch_f = scce_loss(y, logits[0], num_classes)
        loss_per_sample_f[tidx.numpy()] = loss_per_sample_batch_f

    return loss_per_sample_f

def createExperimentNameFromParams(params):

    exp_name = ''
    fields = ['DATASET', 'NUM_EPOCHS', 'BATCH_SIZE', 'TYPE']

    for f in fields:
        if f == fields[-1]:
            exp_name = exp_name + '{0}_{1}'.format(f, params[f])
        else:
            exp_name = exp_name + '{0}_{1}_'.format(f, params[f])

    return exp_name