import keras
import numpy as np
from sklearn import cluster, metrics
import matplotlib.pyplot as plt

# mnist = keras.datasets.mnist
# (train_x, train_y), (test_x, test_y) = mnist.load_data()


def infer_cluster_labels(kmeans, actual_labels):
    """ Infer cluster labels from clusters and actual labels

    Arguments:
        kmeans: the fitted kmeans object
        actual_labels: the correct labels for each data (y from the datasets)
    Returns:
        inferred_labels: a dictionary containing {predicted_label: [image_numbers]}
    """

    inferred_labels = {}

    for i in range(kmeans.n_clusters):
        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

        #print(labels)
        #print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))

    return inferred_labels


def infer_data_labels(X_labels, cluster_labels):
    """ Convert cluster-predicted label numbers into actual labels

    Arguments:
        X_labels: list of cluster-predicted labels
        cluster_labels: inferred_labels from infer_labels
    Returns:
        predicted_labels: mapping between cluster labels and x_labels
    """
    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key

    return predicted_labels

def fit(x_train, train_y):
    # flatten image data using list comprehension
    train_x_flat = [[item for sublist in image for item in sublist] for image in x_train]

    # 1024 clusters (to avoid mislabelling)
    k = 1024
    kmeans = cluster.MiniBatchKMeans(n_clusters = k, init_size=k*3, random_state=0)
    kmeans.fit(train_x_flat)

    # convert flattened centroid images back into full-size
    centroids = kmeans.cluster_centers_
    images = centroids.reshape(k, 28, 28)

    # calculate cluster labels
    cluster_labels = infer_cluster_labels(kmeans, train_y)

    # find different numbered clusters and sort into numerical order
    images_to_show = []
    titles = []
    for i in range(10):
        titles.append(i)
        images_to_show.append(images[cluster_labels[i][0]])

    if 'plot' in kwargs:
        # plot (some) centroids
        fig, axs = plt.subplots(5, 2, figsize = (20, 20))
        plt.gray()

        for i, ax in enumerate(axs.flat):
            ax.set_title(titles[i])
            ax.matshow(images_to_show[i])
            ax.axis('off')

        plt.show()

    return kmeans,

def predict(test_x, args):
    kmeans = args[0]
    # flatten image data using list comprehension
    test_x_flat = [[item for sublist in image for item in sublist] for image in test_x]

    # calculate and print accuracy
    test_clusters = kmeans.predict(test_x_flat)
    predicted_labels = infer_data_labels(test_clusters, cluster_labels)
    print(f'Accuracy: {metrics.accuracy_score(test_y, predicted_labels)}')
    return predicted_labels
