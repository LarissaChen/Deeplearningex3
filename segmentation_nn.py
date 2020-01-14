"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        self.num_classes = num_classes

        # use vgg as the pre-trained model
        self.pretrained = models.vgg11_bn(pretrained=True).features[0:22]  # feature of vgg16: 512*15*15

        # initialize layers for vgg11
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1)  # 1*1*4096
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        # self.conv2 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)
        # self.conv3 = nn.Conv2d(in_channels=4096, out_channels=num_classes, kernel_size=1)

        # # use resnet as the pre-trained model
        # prenet = models.resnet18(pretrained=True)
        # self.pretrained = nn.Sequential(*list(prenet.children())[:-2])
        #
        # # initialize layers for resnet
        # self.conv1 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=8)  # 1*1*4096
        # self.relu = nn.ReLU(inplace=True)  # What does 'inplace' mean?
        # self.dropout = nn.Dropout(p=0.5)
        # self.conv2 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)
        # self.conv3 = nn.Conv2d(in_channels=4096, out_channels=num_classes, kernel_size=1)


        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        # size of x: 10, 3, 240, 240
        input_size = x.size()[2:]  # what should the input_size be like?
        # print('input size', x.size())
        output = self.pretrained(x)  # vgg11:10, 256, 7, 7, res: 10, 1000
        # print('pretrained finished')
        print('pretrained output size', output.size())

        #fcn
        # # for resnet
        # output = torch.unsqueeze(output, 2)
        # output = torch.unsqueeze(output, 2)
        # # print('unsqueeze output size', output.size())

        output = self.conv1(output)
        output = self.relu(output)
        output = self.dropout(output)
        print('frist conv', output.size())

        # output = self.conv2(output)
        # output = self.relu(output)
        # output = self.dropout(output)
        # # print('second conv finished')
        #
        # output = self.conv3(output)
        # # print('third conv finished')

        # output = F.interpolate(output, input_size, mode='bilinear')
        # print('output size', output.size())
        output = F.upsample(output, input_size, mode='bilinear')
        # output = F.upsample(output, input_size, mode='bilinear').contiguous
        print(output.size())
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
