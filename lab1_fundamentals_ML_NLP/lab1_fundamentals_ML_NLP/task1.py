import numpy as np
from matplotlib import pyplot as plt
import PIL
from os import listdir
from os.path import isfile, join

# Task 1.1.1
imagefilenames = [f for f in listdir("./our_dataset") if isfile(join("./our_dataset", f))]
images = []

for imagename in imagefilenames:
    with PIL.Image.open("./our_dataset/" + imagename) as im:
        im = im.resize((30,30))
        image_array = np.array(im)
        images.append(image_array)
        x, y = image_array.shape[:2]

images = np.array(images)

# Task 1.1.2
def PlotSample(index: int, images):
    plt.imshow(images[index])
    plt.axis('off')
    plt.show()

# Task 1.1.3

images_flat= images.reshape(images.shape[0], -1)

def plotImage(X):
    plt.figure(figsize=(1.5, 1.5))
    plt.imshow(X.reshape(30,30,3))
    plt.axis('off')
    plt.show()
    plt.close()


images_norm = images_flat/ 255

images_norm = images_norm - images_norm.mean(axis=0)

cov = np.cov(images_norm, rowvar=False)

U,S,V = np.linalg.svd(cov)

epsilon = 0.1
images_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(images_norm.T).T

images_ZCA_rescaled = (images_ZCA - images_ZCA.min()) / (images_ZCA.max() - images_ZCA.min())

plotImage(images_flat[12, :])
plotImage(images_ZCA_rescaled[12, :])


