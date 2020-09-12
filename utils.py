import torch.nn.functional as F
import torch
import torch.nn as nn

## 这个就是用1*1的卷积实现linear的功能。
class FC(nn.Module):
    def __init__(self,input_channels,output_channels,activations,
                 kernel_size=(1,1),stride=1,padding=0,use_bias=True):
        super(FC, self).__init__()

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias

        self.output_channels = output_channels
        self.activations = activations

        if isinstance(self.output_channels,int):
            self.output_channels = [self.output_channels]
            self.activations = [self.activations]
        elif isinstance(self.output_channels,tuple):
            self.output_channels = list(self.output_channels)
            self.activations = list(self.activations)

        assert type(self.output_channels)==list

        conv_list = []
        for output_channel in self.output_channels:
            conv_list.append(nn.Conv2d(self.input_channels,
                      output_channel, self.kernel_size, self.stride,
                      self.padding, bias=self.use_bias))

            self.input_channels = output_channel


        self.conv2d_modulelist = nn.ModuleList(conv_list)

        self.bn = nn.BatchNorm2d(self.input_channels)



    def forward(self, x):
        x = x.permute(0,3,1,2)
        for i,l in enumerate(self.conv2d_modulelist):
            if self.activations[i]!=None:
                x = l(x)
                x = self.bn(x)
                x = self.activations[i](x)
            else:
                x = l(x)

        return x


#####测试一下FC
if __name__ == '__main__':
    x = torch.randn((10,12,325,1))
    input_channel = x.shape[-1]
    layer1 = FC(input_channel,[64,64],[F.relu,None])
    output = layer1(x)
    print(output.shape)


        
