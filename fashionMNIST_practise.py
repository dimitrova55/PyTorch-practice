import torch
from torch.utils.data import Dataset
import torch.utils.data
import torchvision
from torchvision.transforms import ToTensor, Lambda

import matplotlib.pylab as plt
import pandas as pd
import os

class CustomImageDataset(Dataset):
    """the labels.csv looks like:
        tshirt1.jpg, 0
        tshirt2.jpg, 0
        ......
        ankleboot999.jpg, 9
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = torchvision.io.read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        
        
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output
    
    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Prediction and loss
        predict = model(X)
        loss = loss_fn(predict, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():
        for X, y in dataloader:
            predict = model(X)
            test_loss += loss_fn(predict, y).item()
            correct += (predict.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct) :>0.1f}%, Avg. loss: {test_loss:>8f} \n")
        
        
# configure device
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

""" All TorchVision datasets have two parameters -transform to modify the features and target_transform to modify the labels
The FashionMNIST features are in PIL Image format, and the labels are integers.
For training, we need the features as normalized tensors (ToTensor), and the labels as one-hot encoded tensors (Lambda).
ToTensor converts a PIL image or NumPy ndarray into a FloatTensor. and scales the image's pixel intensity values in the range [0., 1.]

"""
# Download dataset
data_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=ToTensor(), download=True)
                                            #    target_transform=Lambda(lambda y: torch.zeros(
                                            #        10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)))
data_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=ToTensor())

# Visualize dataset
labels_map = {}
for i in range(len(data_train.classes)):
    labels_map[i] = data_train.classes[i]
print(labels_map)

figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(data_train), size=(1,)).item()
    img, label = data_train[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# Load the data
data_train = torch.utils.data.DataLoader(data_train, batch_size=64, shuffle=True)
data_test = torch.utils.data.DataLoader(data_test, batch_size=64, shuffle=True)

# Iterate through the data
train_features, train_labels = next(iter(data_train))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label; {label}")


# create an instance of NeuralNetwork
model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device) # one image of size 28x28
output = model(X)
predict_output = torch.nn.Softmax(dim=1)(output)
y_predict = predict_output.argmax(1)
print(f"Predict class: {y_predict}")

# Optimization: SGD: Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# Loss function
loss_fn = torch.nn.CrossEntropyLoss()

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------------------")
    train_loop(data_train, model, loss_fn, optimizer)
    test_loop(data_test, model, loss_fn)
    
print("Done!")

"""Save the model"""
model = torchvision.models.vgg16(weights='IMAGENET1K_V1')
# PyTorch models store the learned parameters in an internal state dictionary, called state_dict
torch.save(model.state_dict(), 'model_fashion_mnist.pth')

"""Load the model"""
# To load model weights, you need to create an instance of the same model first, 
# and then load the parameters using load_state_dict() method.
# model = torchvision.models.vgg16()
# model.load_state_dict(torch.load('model_fashion_mnist.pth'))
# model.eval()