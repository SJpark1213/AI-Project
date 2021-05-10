import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

model_path = './model_pth/vgg16_bn-6c64b313.pth'

LR = 0.0005  # learning rate
EPOCH = 10  # EPOCH
CLASSES = ('crack', 'normal','Include Resin Bleed Crack')


class VGG(nn.Module):
    def __init__(self, features, num_classes=3):  # num_class
        super(VGG, self).__init__()  # pytorch nn.Module nn.Module的__init__
        self.features = features 
        self.classifier = nn.Sequential(  
                
#            nn.Linear(512*20*15, 1000),
#            nn.ReLU(inplace=True),
#            nn.Dropout(),
#
#            nn.Linear(1000, 1000),
#            nn.ReLU(inplace=True),
#            nn.Dropout(),
#
#            nn.Linear(1000, num_classes))
        
            nn.Linear(512*20*15, num_classes))
        
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'] 


def make_layers(cfg, batch_norm=False):
    
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M': 
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
           
            conv2d = nn.Conv2d(in_channels, v, 3, padding=1)  
            if batch_norm: 
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]   
            in_channels = v  
    return nn.Sequential(*layers)  
    

def vgg16(**kwargs):
    model = VGG(make_layers(cfg, batch_norm=True), **kwargs)  # batch_norm
    # model.load_state_dict(torch.load(model_path))  
    return model

        
def process(loader):
    fig = plt.figure(figsize=(100,100))
    rows = 1
    columns = 4
    i=1
    for idx, (inputs_cpu, labels_cpu) in enumerate(loader):
        img = inputs_cpu.numpy()
        img = np.transpose(img,(1,2,0))
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        i +=1
        if i ==4:
            break
    plt.show()    

def imshow(img):
    img = img /2 + 0.5 # unnormalize
    np_img = img.numpy()
    # plt.imshow(np_img)
    plt.imshow(np.transpose(np_img, (1,2,0)))
    
    print(np_img.shape)
    print((np.transpose(np_img,(1,2,0))).shape)
    
    
          
def getData(): 
    # transforms.Compose([...])
    transform = transforms.Compose([
#        transforms.CenterCrop(480),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                             std=[1, 1, 1])])  # input[channel] =(input[channel] - mean[channel])/std[channel]
    trainset = tv.datasets.ImageFolder(root='C:/Users/User/Desktop/VGG16/dataset/trainset', transform=transform)  #CIFAR10
    testset = tv.datasets.ImageFolder(root='C:/Users/User/Desktop/VGG16/dataset/testset', transform=transform)  # CIFAR10
    
    img1 = trainset[0][0].numpy()
    #plt.imshow(np.transpose(img1,(1,2,0)))
    img2 = testset[0][0].numpy()
    #plt.imshow(np.transpose(img2,(1,2,0)))

    train_loader = DataLoader(trainset, batch_size=4, shuffle=True)  
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)
          
    return train_loader, test_loader

weight_scaling = [0.45,0.45,0.1]
weight_scaling = torch.Tensor(weight_scaling)


def train():
    """创建网络，并开始训练"""
    trainset_loader, testset_loader = getData()
    net = vgg16().cuda()  # vgg16
    net.train()
    print(net)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=weight_scaling).cuda()  #CrossEntropyLoss, softmax，softmax
    optimizer = torch.optim.Adam(net.parameters(), lr=LR) #Adam

    # Train the model
    for epoch in range(EPOCH):
        true_num = 0.
        sum_loss = 0.
        total = 0.
        accuracy = 0.
        train_loss = []
        cnt = 0
        for step, (inputs_cpu, labels_cpu) in enumerate(trainset_loader):
            inputs = inputs_cpu.cuda()
            labels = labels_cpu.cuda()
            output = net(inputs)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Adam
            
            _, predicted = torch.max(output, 1)  # predicted 
            sum_loss += loss.item()
            train_loss.append(loss.item())
            total += labels.size(0)
            accuracy += (predicted == labels).sum()
            cnt += 1
            print("epoch %d | step %d: loss = %.4f, the accuracy now is %.3f %%." % (epoch, step, sum_loss/(step+1), 100.*accuracy.cpu().numpy()/total))
       
        acc = test(net, testset_loader)
        print("")
        print("___________________________________________________")
        print("epoch %d : training accuracy = %.4f %%" % (epoch, 100 * acc))
        print("---------------------------------------------------")
        x=np.linspace(1,cnt,cnt)
        y=train_loss
        plt.plot(x,y)
        plt.show()


    print('Finished Training')
    return net


def test(net, testdata):  
    
    correct, total = .0, .0
       
    for inputs_cpu, labels_cpu in testdata:
        cnt += 1
        inputs = inputs_cpu.cuda()
        labels = labels_cpu.cuda()
        net.eval()  # training和test/evaluation, dropout。
                    # net.eval()evaluation
        with torch.no_grad():
            
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            

        #print(predicted)
    # net.train()
    return float(correct.cpu().numpy()) / total

if __name__ == '__main__':
    net = train()
    

def show(img):
    npimg = img.numpy()
    fig = plt.figure(figsize=(25,5))
    plt.imshow(np.transpose(npimg,(1,20)), interpolation = 'nearest')
    