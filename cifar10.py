import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images/255.0, test_images/255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# # I tried my best to use the VGG-19 Architecture
# model = models.Sequential([
#     layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')  
# ])

# model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy'])

# history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# model.save("cifar10_vgg19type_convolutionalNeuralNetwork.h5")
# print("done")
cifar10cnn = models.load_model("cifar10_vgg19type_convolutionalNeuralNetwork.h5")
# test_loss, test_acc = cifar10cnn.evaluate(test_images, test_labels, verbose=2)

predictions = cifar10cnn.predict(test_images)

test_labels = np.array(test_labels)
print(test_images[0].shape)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i][0]])
    plt.title("Prediction: " + class_names[np.argmax(predictions[i])])
    plt.show()