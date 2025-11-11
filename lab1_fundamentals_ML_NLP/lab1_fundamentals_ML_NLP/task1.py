import numpy as np
from matplotlib import pyplot as plt
import PIL
from os import listdir
from os.path import isfile, join

# Task 1.1.1
imagefilenames = [f for f in listdir("./our_dataset") if isfile(join("./our_dataset", f))]
images = []
samples = []

for imagename in imagefilenames:
    with PIL.Image.open("./our_dataset/" + imagename) as im:
        im = im.resize((30,30))
        image_array = np.array(im)
        images.append(image_array)
        x, y = image_array.shape[:2]
        samples.append([len(imagefilenames), x, y, 3])

samples = np.array(samples, dtype=int)
images = np.array(images)

# Task 1.1.2
def PlotSample(index: int, images):
    plt.imshow(images[index])
    plt.axis('off')
    plt.show()

# Task 1.1.3


