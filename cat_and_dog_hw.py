import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

import matplotlib.pyplot as plt
import numpy as np

# Load, Rescale the dataset
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, horizontal_flip=True, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory('PetImages\\training_set', target_size=(128,128), batch_size=32, class_mode='binary', subset='training')
valid_data = train_datagen.flow_from_directory('PetImages\\training_set', target_size=(128,128), batch_size=32, class_mode='binary', subset='validation')
test_data = test_datagen.flow_from_directory('PetImages\\test_set', target_size=(128,128), batch_size=32, class_mode='binary')

# print(train_data.image_shape)

# Visualize the dataset
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(next(iter(train_data[i][0])), cmap='gray')
plt.show()

# Build the model
model = keras.models.Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(units=128, activation='relu'),
    Dense(units=1, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, validation_data=valid_data, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(valid_data)
print(f'Loss: {loss:.3f}, Accuracy: {accuracy:.3f}')

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
fig.savefig('loss_accuracy_pets.jpg')

# make predictions using the model
predictions = model.predict(test_data)
print(np.argmax(predictions, axis=1)) # prediction of the 1st 5 samples

# Visualize the 1st 10 images and labels of the test dataset
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_data[i], cmap=plt.cm.binary)
    # print(test_data[i])
    # plt.xlabel(test_data[i])
plt.show()