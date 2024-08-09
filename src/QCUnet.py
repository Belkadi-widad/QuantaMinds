import os
import numpy as np
import torch
from torch import nn, optim
from torchsummary import summary
import torch.nn.functional as F

from pennylane import *

from classical_unet import *
from train import *
# from QCNN_circuit import *
from unitary import * 
from QCUBottleNeck import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

max_qubits=16
if use_cuda is True:
    dev = qml.device("lightning.gpu", wires=max_qubits)
else:
    dev = qml.device("default.qubit", wires=max_qubits)


## QCNN circuit  
@qml.qnode(dev)
def circuit(inputs, weights, n_qubits, unitary =None ):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    if unitary is None: 
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    elif unitary ==   'U_TTN':
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation= RY)

    else: 
        unitary(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]



def calculate_padding(window_size, stride, height, width):
    if stride == 1:
        pad_height = max(window_size - 1, 0)
        pad_width = max(window_size - 1, 0)
    else:
        pad_height = (window_size - stride + height % stride) % window_size
        pad_width = (window_size - stride + width % stride) % window_size
    return pad_height, pad_width

# we can add the QCNN circuit as a parameter 

def quanv(image, weights, window_size=2, stride = 2, unitary = U_TTN):
    """Convolves the input image with many applications of the same quantum circuit."""
    # PyTorch models generally require a 4D input tensor with the
    # dimensions - (batch size, channels, height, width)
    # The input image shape is (batch_size, 1, height, width)
    # The output shape should be (batch_size, n_channels, height//window_size, width//window_size)
    
    batch_size, _, height, width = image.shape
    out_height = height // stride
    out_width = width // stride

    pad_height, pad_width = calculate_padding(window_size, stride, height, width)

    # Add padding to the image
    image = F.pad(image, (0, pad_width, 0, pad_height), mode='constant', value=0)
    l_image = image.detach().cpu().numpy()
    l_weights = weights.detach().cpu().numpy()

    padded_height, padded_width = height + pad_height, width + pad_width
    out_height = (padded_height - window_size) // stride + 1
    out_width = (padded_width - window_size) // stride + 1
 
    n_qubits= window_size * window_size

    out = np.zeros((batch_size, n_qubits, out_height, out_width))

    for i in range(batch_size):
        # Loop over the coordinates of the top-left pixel of window_size x window_size squares
       for j in range(0, padded_height - window_size + 1, stride):
            for k in range(0, padded_width - window_size + 1, stride):
                # Extract the pixels in the window and process them with a quantum circuit
                window_pixels = [l_image[i, 0, j + m, k + n] * np.pi for m in range(window_size) for n in range(window_size)]
                q_results = circuit(window_pixels, l_weights, n_qubits = n_qubits, unitary= unitary)
                # Assign expectation values to different channels of the output pixel (j//window_size, k//window_size)
                for c in range(n_qubits):
                    out[i, c, j // window_size, k // window_size] = q_results[c]

    return out


# Custom nn.Module class for handling the quantum convolution
class QuantConv(nn.Module):

    # The number of layers in the BasicEntanglerLayers instance in
    # the quantum convolver

    def __init__(self, window_size, num_layers, n_qubits, stride, unitary= U_TTN ):
        super(QuantConv, self).__init__()
        # Initialise and register weights
        # weights have shape (LAYERS, n_qubits) where layers is the
        # number of layers in the BasicEntanglerLayers
        self.num_layers= num_layers
        self.window_size=  window_size
        self.n_qubits = n_qubits
        self.stride= stride
        self.unitary = unitary
        if self.n_qubits is None: 
            self.n_qubits = window_size*window_size
        if self.unitary is None or unitary == 'U_TTN': 
            self.weights = nn.Parameter(
            torch.from_numpy(np.random.uniform(
                0, np.pi, (self.num_layers, self.n_qubits))))
        else: 
            params= initialize_params(unitary)
            self.weights = nn.Parameter(
                torch.from_numpy(params))
            # self.weights = nn.Parameter(
            # torch.from_numpy(np.random.uniform(
            #     0, np.pi, self.n_qubits)))
    def forward(self, input):
        expectation_z = quanv(input, self.weights, window_size=self.window_size, stride = self.stride, unitary= self.unitary)
        x = torch.tensor(expectation_z)
        return x



#s U-Net implementation is available from
# https://github.com/milesial/Pytorch-UNet
# under GNU General Public License v3.0
#
# Copyright (C) 2017 https://github.com/milesial


class QDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,  in_channels, out_channels,  window_size = 2,stride = 2,   
                 mid_channels=None, unitary= U_TTN ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.qconv= QuantConv(n_qubits=in_channels, num_layers= 1, window_size = window_size, stride= stride, unitary=unitary )
        self.conv_2d= nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x=  self.qconv(x)
        x = x.float()
        x=  self.conv_2d(x)
        x=  self.batchNorm(x)
        x=  self.relu(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,   quantum = False, window_size=2, stride= 2, unitary= U_TTN ):
        super().__init__()
        if quantum: 
            convLayer= QDoubleConv(in_channels, out_channels, 
                                   window_size = window_size, 
                                   stride = stride, unitary = unitary)
        else: 
            convLayer= DoubleConv(in_channels, out_channels)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            convLayer
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class QCUNet(nn.Module):
    def __init__(self, n_channels, n_classes, quantum_layers= [True, True, True], 
                 unitary = U_TTN, bilinear=True):
        super(QCUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.Qbottleneck = quantum_layers[2]
        if quantum_layers[0]:
            self.inc = QDoubleConv(in_channels=4, out_channels=9,  window_size = 2, stride= 1, 
                                   unitary= unitary)
        else: 
            self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(in_channels = 9, out_channels = 64, window_size = 3, quantum = quantum_layers[1], 
                          stride= 1)
        self.down2 = Down(in_channels = 64, out_channels = 128, window_size = 3, 
                          quantum = False)
        self.down3 = Down(in_channels = 128, out_channels = 256,  quantum = False)
        self.down4 = Down(256, 512, quantum = False)
        factor = 2 if bilinear else 1
        self.down5 = Down(512, 1024 // factor, quantum = False)
        
        if quantum_layers[2]:
            self.qcnn = QBottleneck(4,unitary=unitary).to(device)
            
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 9, bilinear)
        self.up5 = Up(18, 9, bilinear)
        self.outc = OutConv(9, n_classes)

   
    def forward(self, x):
            x1 = self.inc(x)  
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            if self.Qbottleneck: 
                # Apply QCNN layer in the bottleneck
                batch_size, channels, height, width = x6.shape
                x6_flatten = torch.flatten(x6, start_dim=2)  # Flatten the spatial dimensions (except batch size)
                qcnn_output = self.qcnn(x6_flatten)
                x6 = qcnn_output.view(batch_size, 512, height, width)  # Adjust as per actual output dimensions
                
            x = self.up1(x6, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)
            logits = self.outc(x)

            return logits



