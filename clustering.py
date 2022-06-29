import os
import utils
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pandas as pd
import time


def clustered_pixels(x_fit, pixels):
    labels = x_fit.predict(pixels)
    res = x_fit.cluster_centers_[labels]
    return res


def kmeans_image(image):
    image_data = np.array(image.getdata()) / 255.0
    kmeans = KMeans(n_clusters=3, random_state=0)
    x_fit = kmeans.fit(image_data)
    return image_data, x_fit


def main():
    image_paths = os.listdir("./images")
    start = time.time()
    images = [Image.open(os.path.join("images", path)) for path in image_paths]
    print("Images loaded: ", time.time() - start)
    compressed_data = []
    for image in images:
        start = time.time()
        width, height = image.size
        image_data, x_fit = kmeans_image(image)
        clust_pixels = clustered_pixels(x_fit, image_data)
        print(clust_pixels.shape)
        clust_image = np.reshape(clust_pixels, (height, width, 3))
        # compressed_data.append(clust_image)
        # plt.imshow(clust_image, interpolation="nearest")
        # plt.show()
        print("Took: ", time.time() - start, "seconds")


if __name__ == "__main__":
    main()
