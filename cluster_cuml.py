import os
import numpy as np
from PIL import Image
from cuml import KMeans
from matplotlib import pyplot as plt
import pandas as pd
import time


def clustered_pixels(x_fit, pixels):
    """
    It takes the pixels of an image and the fitted model, and returns the pixels of the image with the
    colors replaced by the colors of the clusters

    :param x_fit: the KMeans object that has been fit to the data
    :param pixels: the image data, which is a 3-dimensional array of size [h, w, 3]
    :return: The cluster centers of the pixels.
    """
    labels = x_fit.predict(pixels)
    res = x_fit.cluster_centers_[labels]
    return res


def kmeans_image(image):
    """
    It takes an image, converts it to a numpy array, and then uses the KMeans algorithm to cluster the
    image into 3 clusters

    :param image: the image to be processed
    :return: The image data and the fit of the kmeans algorithm
    """

    image_data = np.array(image.getdata()) / 255.0
    kmeans = KMeans(n_clusters=3, random_state=0)
    x_fit = kmeans.fit(image_data)
    return image_data, x_fit


def main():

    image_paths = os.listdir("./images")
    images = [Image.open(os.path.join("images", path)) for path in image_paths]

    for image in images:
        start = time.time()
        image.thumbnail((200, 200))
        width, height = image.size

        image_data, x_fit = kmeans_image(image)
        clust_pixels = clustered_pixels(x_fit, image_data)
        clust_image = np.reshape(clust_pixels, (height, width, 3))
        print('Took: ', time.time() - start)
        #plt.imshow(clust_image, interpolation="nearest")
        #plt.show()


if __name__ == "__main__":
    main()
