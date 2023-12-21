import keras
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.datasets import mnist
from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt

# load the dataset
"""
x_train: uint8 NumPy array of grayscale image data with shapes (60000, 28, 28). 
    Pixel values range from 0 to 255.
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalization of the images, converting to values between 0 and 1 
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# axis=-1 : adds a new axis at the back (height, width and channel)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# one-hot encoding of labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# model structure
model = keras.models.Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Additional 2 layers
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Loss; {loss:.3f}, Accuracy: {accuracy:.3f}')

# graph of loss and accuracy 
fig = plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss function over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Draw accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# save the image
fig.savefig('loss_accuracy_model2.jpg')

# make predictions using the model
predictions = model.predict(x_test[:5])
print(np.argmax(predictions, axis=1)) # prediction of the 1st 5 samples

# Visualize the 1st 10 images and labels of the test dataset
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    # plt.xlabel(y_test[i])
plt.show()

