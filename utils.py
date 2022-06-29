import matplotlib.pyplot as plt


def plot(image):
    if image.getdata().max() > 1.0:
        plt.imshow(image / 255, interpolation="nearest")
        plt.show()
    else:
        plt.imshow(image, interpolation="nearest")
        plt.show()
