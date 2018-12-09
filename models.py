## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # 1 input image channel (grayscale), 32 output channels/feature maps
        # 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # second conv layer: 32 inputs, 64 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output tensor will have dimensions: (64, 106, 106)
        # after another pool layer this becomes (64, 53, 53); *.5 is rounded down
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third conv layer: 64 inputs, 128 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (53-5)/1 +1 = 49
        # the output tensor will have dimensions: (64, 49, 49)
        # # after another pool layer this becomes (128, 24, 24); *.5 is rounded down
        self.conv3 = nn.Conv2d(64, 128, 5)

        # 128 outputs * the 49*49 filtered/pooled map size
        self.fc1 = nn.Linear(128*49*49, 136*5)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
            
        # finally, 136 outputs (2 for each of the 68 keypoints)
        self.fc2 = nn.Linear(5 * 136, 2*68)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

# Net1 cannot get under 0.8 loss on training set, the model is not powerful enough
# let's try adding more layers
# steal some ideas from alexnet
class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        
        # 1 input image channel (grayscale), 64 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (224-3)/1 +1 = 222
        # the output Tensor for one image, will have the dimensions: (64, 222, 222)
        # after one pool layer, this becomes (64, 111, 111)
        self.conv1 = nn.Conv2d(1, 64, 3)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # second conv layer: 64 inputs, 128 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (111-5)/1 +1 = 107
        # the output tensor will have dimensions: (128, 107, 107)
        # after another pool layer this becomes (128, 53, 53); *.5 is rounded down
        self.conv2 = nn.Conv2d(64, 128, 5)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third conv layer: 128 inputs, 256 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (53-3)/1 +1 = 50
        # the output tensor will have dimensions: (256, 50, 50)
        self.conv3 = nn.Conv2d(128, 256, 3)
        
        # Fourth conv layer: 256 inputs, 256 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (50-3)/1 +1 = 48
        # the output tensor will have dimensions: (256, 48, 48)
        self.conv4 = nn.Conv2d(256, 256, 3)
        
        # Fifth conv layer: 256 inputs, 256 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (48-3)/1 +1 = 46
        # the output tensor will have dimensions: (256, 46, 46)
        # after another pool layer this becomes (256, 23, 23); *.5 is rounded down
        self.conv5 = nn.Conv2d(256, 256, 3)
        
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # 256 outputs * the 46*46 filtered/pooled map size
        self.fc1 = nn.Linear(256*23*23, 23*23)

        self.fc2 = nn.Linear(23*23, 136*2)
        
        # dropout with p=0.4
        self.fc2_drop = nn.Dropout(p=0.4)
            
        # finally, 136 outputs (2 for each of the 68 keypoints)
        self.fc3 = nn.Linear(136*2, 2*68)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # also use Alexnet for inspiration

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool5(F.relu(self.conv5(x)))
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x