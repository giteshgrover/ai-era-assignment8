import torch
import torch.nn as nn
import torch.nn.functional as F

receptive_field_max_n_j_in = [1, 1]

# A BasicBlock class that implements the residual block structure with:
## Two convolutional layers with batch normalization. Where, the first convolution is a depthwise convolution and second one is a normal convolution.
## A skip connection (shortcut)
## ReLU activation
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        k_size = 3

        #  Replaced the single conv1 layer with two:
        ## conv1_depthwise: A depthwise convolution where groups=in_planes means each input channel is convolved separately. Stride is what passed in.
        ## conv1_pointwise: A 1x1 convolution that mixes the channels
        # Modified the forward pass to use these two convolution operations in sequence
        self.conv1_depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=3, 
                                        stride=stride, padding=1, groups=in_planes, bias=False)
        self.conv1_pointwise = nn.Conv2d(in_planes, planes, kernel_size=1, 
                                        stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        receptive_field_max_n_j_in[0] = receptive_field_max_n_j_in[0] + (k_size-1) * receptive_field_max_n_j_in[1] 

        # Regular Convolution with stride = 1 and padding & dialation = passed in val
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        receptive_field_max_n_j_in[0] = receptive_field_max_n_j_in[0] + (k_size-1) * dilation * receptive_field_max_n_j_in[1] # Adding dilation in RF calculation. Check if its correct?
        receptive_field_max_n_j_in[1] = receptive_field_max_n_j_in[1] * stride  
        print(f"****Receptive Field : {receptive_field_max_n_j_in[0]}")

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            # If the output of the residuals doesn't match the skip, use 1x1
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1_depthwise(x)
        out = self.conv1_pointwise(out)
        out = F.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # Add skip
        out = F.relu(out) # Relu after adding the skip (Resnet v2)
        return out

# A ResNet class that:
## Takes the block type and number of blocks per layer as parameters
## Implements the full ResNet architecture
## Uses an initial convolutional layer followed by 4 layer groups
## Ends with average pooling and a fully connected layer
#
# This implementation is specifically adapted for CIFAR10:
# Uses 3x3 initial convolution instead of 7x7 (since CIFAR10 images are smaller than ImageNet)
# Removes initial max pooling layer
# Uses smaller stride in the first layer
# Final average pooling is 4x4 instead of 7x7
# The model will accept 32x32x3 images (CIFAR10 format) and output 10 classes.
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 32
        block = BasicBlock

        # Input: 3 x 32 x 32, Output: 32 x 32 x 32, RF: 3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Layers with multiple blocks with each block has two convolution and a skip
        # Layer 1 - input 32 x 32 x 32, output: 32 x 16 x 16, RF (rf-in + 2 * 2 * 3): 15, 11, 9,... 
        self.layer1 = self._make_layer(block, planes=32, block_strides=[1, 1, 2], dilations=[1, 1, 1])  
        # Layer 2 - input 32 x 16 x 16, output: 32 x 8 x 8, RF (rf-in + 4 * 2 * 3): 39, 35, 31, ...
        self.layer2 = self._make_layer(block, planes=32, block_strides=[1, 1, 2], dilations=[1, 1, 1]) 
        # Layer 3 - input 32 x 8 x 8, output: 32 x 4 x 4, RF (rf-in + 8 * 2 * 3): 87, 79, 71, ...
        self.layer3 = self._make_layer(block, planes=32, block_strides=[1, 1, 2], dilations=[1, 1, 1]) 
        # Layer 4 - input 32 x 4 x 4, output: 32 x 4 x 4, RF: 
        self.layer4 = self._make_layer(block, planes=32, block_strides=[1, 1, 1], dilations=[1, 2, 1])
        self.linear = nn.Linear(32*block.expansion, num_classes)

    def _make_layer(self, block, planes, block_strides, dilations):
        # A Layer made of multiple blocks where the first CN in block will have stride of the 'stride' value passed in 
        # The 2nd CN in 1st block will have stride of 1. Also, all the following up blocks will have stride of '1'
        # strides = [stride, 1, 1, ...]
        # strides = [1]*(num_blocks-1) + [2]
        layers = []
        # print(f"MakeLayer passed in stride value {stride}")
        # print(f"strides {strides}")
        for idx, stride in enumerate(block_strides):
            layers.append(block(self.in_planes, planes, stride, dilations[idx]))
            self.in_planes = planes * block.expansion # Change the input for next layer to the output of current layer
        # print(f"layers {layers}")
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial Convolution
        out = F.relu(self.bn1(self.conv1(x))) 
        # n Layers with n.m blocks in each layer
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # GAP
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # Fully Connected layer (TODO Optional and can be changed )
        out = self.linear(out)
        return out

# A ResNet18() function that returns a ResNet-18 model configured for CIFAR10 (10 classes)
def ResNet18():
    model = ResNet()
    print(f"Receptive Field Max: {receptive_field_max_n_j_in[0]}")
    return model
