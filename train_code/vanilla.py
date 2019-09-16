import sys
sys.path.append("../")
import logging
logging.basicConfig(format='[%(asctime)s %(levelname)s] \n %(message)s', level=logging.INFO)
import pickle
import torch.optim as optim


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model.classifier_1 import Classifier as Model1



TRAIN_EPOCH = 64
LR = 1e-3
MOMENTUM = 0.9


if __name__ == "__main__":
    # init classifier
    m = Model1()
    logging.info(m)
    
    # load data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    # loss and opt
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(m.parameters(), lr=LR, momentum=MOMENTUM)
    
    # GPU training
    device = torch.device("cuda:0")
    m.to(device)
    for epoch in range(TRAIN_EPOCH):  # loop over the dataset multiple times
        running_loss = []
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = m(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss.append(loss.item())
            if i % 100 == 99:
                logging.info('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, sum(running_loss) / len(running_loss)))
                running_loss = []
        logging.info('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, sum(running_loss) / len(running_loss)))
    logging.info('Finished Training')
    
    # save model
    pickle.dump(m.state_dict(), open('../saved/vanilla', 'wb'))
    logging.info('Model Saved')
    
    # init new model and load
    new_m = Model1()
    new_m.load_state_dict(pickle.load(open('../saved/vanilla', 'rb')))
    logging.info('Model Loaded')
    
    # evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = new_m(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logging.info('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))