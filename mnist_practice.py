"""
MNIST - Modified National Institute of Standards and Technology 
Handwritten digits recognition
"""
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt



# Configure Device
device = torch.device("cude" if torch.cuda.is_available() else 'cpu')

# MNIST Dataset download
# 'root' variable holds the path where we want to store the dataset
train_ds = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_ds = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# Data Loader
train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=100, shuffle=True)
test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=100, shuffle=False)

# Show some examples
examples = iter(test_dl)
example_images, example_targets = next(examples)

for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(example_images[i][0], cmap='gray')
plt.show()

# Bult the neural network
"""image size 28x28 -> input size: 784
    nn.Sequential() wraps the layers in the network
    LogSoftmax function is a logarithm of a Softmax function"""
input_size = 784
hidden_sizes = [128, 64]    # the size of layers 1 & 2
output_size = 10            # classification among 10 digits 0~9

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

print(model)

# Cross-Entrophy Loss Function
loss_fn = nn.CrossEntropyLoss()
images, labels = next(iter(train_dl))
images = images.view(images.shape[0], -1)

log_probs = model(images)   # log probabilities
loss = loss_fn(log_probs, labels)

print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in train_dl:
        # flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        
        # training pass
        optimizer.zero_grad()
        predict = model(images)
        loss = loss_fn(predict, labels)
        
        # backpropagation
        loss.backward()
        
        # optimization of the weights
        optimizer.step()
        
        running_loss += loss.item()
        
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_dl)))
        
        
# Testing
correct_count, all_count = 0, 0
for images, labels in test_dl:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            predict = model(img)
            
        ps = torch.exp(predict)
        probability = list(ps.numpy()[0])
        predict_label = probability.index(max(probability))
        true_label = labels.numpy()[i]
        if(true_label == predict_label):
            correct_count += 1
        all_count += 1
        
print("Number of images tested = ", all_count)
print("\nModel accuracy = ", (correct_count / all_count))
        
# saving the model
torch.save(model, './my_mnist_model.pt')