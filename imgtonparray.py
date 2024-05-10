# Import the necessary libraries
from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import models

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

IMG_PATH = "images\\deer2.jpg"

# load the image and convert into
# numpy array
img = Image.open(IMG_PATH)
img = img.resize((32, 32))
# img.show()
# asarray() class is used to convert
# PIL images into NumPy arrays
arr = asarray(img)
arrd = arr
# print(arr)
arr.shape = (32, 32, 3)
arr = arr/255.0
arr = np.expand_dims(arr, axis=0)

# This image will be displayed
img_display = mpimg.imread(IMG_PATH)

# loading the model
cifar10cnn = models.load_model("cifar10_vgg19type_convolutionalNeuralNetwork.h5")
prediction = cifar10cnn.predict([arr])
plt.title("Prediction: " + class_names[np.argmax(prediction[0])])
plt.imshow(img_display, cmap=plt.cm.binary)
plt.show()

