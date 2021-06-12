#Author: Ian Edwards EDWIAN004
#CSC3022F ML assignment 3
import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

import random
from random import seed
from random import randint
import sys
import argparse
#text for argument parser:
text = 'Classifying handwritten digits with a convolutional neural network built with Pytorch.' 

def main():
    #using MNIST data set from MNIST database. Has 60000 training images and 10000 test images.
    print("Pytorch Output...")

    #transform the data to ensure common dimensions and properties. Creates a tensor which is a image converted to numbers with RGB pixels converted to 0 to 255 scaled to a range of 0 to 1.
    #Set mean and SD of each tensor to 0.5 and 0.5
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

    #load training data from MNIST file in local directory. The 2D structure of the images is preserved with a CNN.
    #The dataset stores the examples as well as the ground truths for the examples.
    trainset = datasets.MNIST('', download=False, train=True, transform=transform)
    valset = datasets.MNIST('', download=False, train=False, transform=transform) #use test set as validation set. Can split data further here. Maybe perform K-fold cross validation?
    #investigate_data(trainset)

    #Use a data loader to make the dataset iterable, giving access to sample data. During training we will pass minibatches of samples in, reshuffle the data at every epoch to prevent overfitting.
    #This functionality is implemented efficiently by a data loader. Data will be fit on trainloader, not valloader - preventing information leak.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    #investigate_loader(trainloader)
    #images will be flattened out in first layer so first layer is 28x28 = 784 neurons
    inputsize = 784
    hiddenlayers = [128, 64]
    #output layer is 10 neurons as there are 10 different types of digits to be classified that are then chosen from using a softmax function.
    outputsize = 10
    model = createModel(inputsize, hiddenlayers, outputsize)

    if (os.path.isfile('./my_mnist_model.pt')):
        model = torch.load('./my_mnist_model.pt')
        print("Model loaded successfully!", model.parameters)
    else:
        trainModel(model, trainloader)

    #use validation set to get model accuracy:
    correct , total = 0, 0
    with torch.no_grad():
        for images, labels in valloader:
            images = images.view(images.shape[0], -1)
            output = model(images)
            for index, i in enumerate(output):
                if torch.argmax(i) == labels[index]:
                    correct +=1
                total +=1
    print(f'accuracy: {round(correct/total, 3)}')

    #saveModel(model, './my_mnist_model.pt')
    inputData = input("Please enter a filepath:")
    while (inputData != "exit"):
        checkImage(valset[inputData][0], model)

def investigate_data(dataset):
    figure = plt.figure(figsize=(6,6))
    for i in range(1, 11):
        index = torch.randint(len(dataset), size=(1,)).item() #Get random tensor and convert to integer to get image index from entire dataset.
        print(index)

        image, label = dataset[index] #retrieve single image-label pair from dataset.
        #print(images.shape) #print image shape and label shape. Expect number of images, followed by resolution.
        #print(labels.shape)
        figure.add_subplot(5, 5, i)
        plt.axis("off")
        plt.title(label)
        plt.imshow(image.squeeze(), cmap='gray_r')
    plt.show()

def investigate_loader(loader):
    #shows that a batch of images from loader is 64 big as specified previously.
    #Each image in each batch is 28x28. The labels batch is 64 big as expected, to match the images.
    images, labels = next(iter(loader))
    print("Image batch shape:", images.size())
    print("or: ", images.shape)
    print("Label batch shape:", labels.size())
    image = images[0].squeeze()
    label = labels[0]
    print("Label:", label)
    plt.imshow(image, cmap="gray")
    plt.show()
    
def createModel(inputsize, hiddenlayers, output_size):
    linear_input = nn.Linear(inputsize, hiddenlayers[0])
    linear_hidden1 = nn.Linear(hiddenlayers[0], hiddenlayers[1])
    linear_hidden2 = nn.Linear(hiddenlayers[1], output_size)
    final_layer = nn.LogSoftmax(dim=1) #ideal for classification problems
    
    #use of ReLu activation functions in between layers because its easy to train with and improve performance.
    model = nn.Sequential(linear_input, nn.ReLU(), linear_hidden1, nn.ReLU(), linear_hidden2, final_layer)
    #print(model)
    return model

def trainModel(model, trainloader):
    #loss function should be a C class which are good for classification problems.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #use Adam optimizer which applies gradient descent - but good optimizer avoids getting stuck in local minima.
    epoch = 10

    for e in range(epoch):
        currentloss = 0
        model.train()
        for images, labels in trainloader:
            #flatten images:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad() #set gradients to 0.
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            currentloss += loss.item()
        print("Epoch: ", e, " : Training Loss = ", currentloss/len(trainloader))
        print(output)
    #print("Training time:")

def checkImage(image, model):
    print(torch.argmax(model(image.view(-1, 784))[0]))
    plt.imshow(image.view(28, 28))
    plt.show()
    

def saveModel(model, path):
    torch.save(model, path) 

def loadModel():
    torch.load('./my_mnist_model.pt')
    
if __name__ == "__main__":
    main()