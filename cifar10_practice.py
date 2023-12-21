import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # define the convolutional and fully-connected layers
        # 1st conv layer: 1 input img channel, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        
    def forward(self, x):
        # max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  
    
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
    
def main():
    
    # create an instance of LeNet class
    model = LeNet()
    print(model)
    
    """
    transforms.ToTensor() converts images loaded by Pillow into PyTorch tensors.
    transforms.Normalize() adjusts the values of the tensor so that their average is zero and their standard deviation is 1.0.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load the data
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_data = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)
    
    test_data = torchvision.datasets.CIFAR10(root='/data', train=False, download=True, transform=transform)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Visualize
    images, labels = next(iter(train_data))
    img_grid = torchvision.utils.make_grid(images)
    imshow(img_grid)
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train
    for epoch in range(2):
        
        running_loss = 0.0
        for i, data in enumerate(train_data, 0):
            imgs, labels = data
            
            # Zeroing the gradients. Gradients are accumulated over a batch; 
            # if we do not reset them for every batch, they will keep accumulating.
            optimizer.zero_grad()
            
            predict = model(imgs)
            # Compute the loss
            loss = loss_fn(predict, labels)
            # Backward pass
            loss.backward()
            # Updating the weights
            optimizer.step()
            
            running_loss +=loss.item()   # accumulated loss
            
            if i % 2000 == 1999:
                print('[%d, %5d] loss %.3f' % (epoch+1, i+1, running_loss / 2000))
                running_loss = 0.0
    
    print("End of training!")
    
    # Test
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_data:
            imgs, labels = data
            predict = model(imgs)
            _, predicted = torch.max(predict.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))        
    
if __name__ == "__main__":
    main()