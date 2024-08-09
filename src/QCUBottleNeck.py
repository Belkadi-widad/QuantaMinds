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



class QBottleneck(nn.Module):
    def __init__(self, n_qubits=4, unitary=U_TTN, num_layers= 1):
        super(QBottleneck, self).__init__()
        self.n_qubits = n_qubits
        self.q_device = qml.device('default.qubit', wires=self.n_qubits)
        self.unitary = unitary
        self.q_node = qml.QNode(self.q_circuit, self.q_device)
        self.num_layers = num_layers
        if self.unitary is None or unitary == 'U_TTN':  
            self.params = nn.Parameter(
            torch.from_numpy(np.random.uniform(
                0, np.pi, (self.num_layers, self.n_qubits))))
        else:
            params= initialize_params(unitary)
            self.params = nn.Parameter(torch.from_numpy(params))
            # self.params = torch.nn.Parameter(torch.randn(self.n_qubits))
        
    def q_circuit(self, inputs, params):
        ## embeddings
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        ## Unitary
        if self.unitary is None: 
            qml.BasicEntanglerLayers(params, wires=range(self.n_qubits))
        elif self.unitary ==   'U_TTN':
            qml.BasicEntanglerLayers(params, wires=range(self.n_qubits), rotation= RY)
        else:
            self.unitary(params, wires=range(self.n_qubits))
        ## Measurement 
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        batch_size, num_channels, num_features = x.shape
        x_np = x.detach().cpu().numpy()  # Convert to NumPy array
        qcnn_output = []

        for i in range(batch_size):
            for j in range(num_channels):
                q_node_output = self.q_node(x_np[i, j], self.params.detach().cpu().numpy())
                qcnn_output.append(q_node_output)

        qcnn_output = np.array(qcnn_output)  # Convert list of arrays to a NumPy array
        qcnn_output = qcnn_output.reshape(batch_size, num_channels, -1)  # Reshape correctly
        qcnn_output = torch.tensor(qcnn_output, device=x.device, dtype=torch.float32)  # Convert back to PyTorch tensor
        return qcnn_output


class UNetWithQBottleNeck(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, unitary = U_TTN):
        super(UNetWithQBottleNeck, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.qcnn = QBottleneck(4,unitary=unitary).to(device)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Apply QCNN layer in the bottleneck
        batch_size, channels, height, width = x5.shape
        x5_flatten = torch.flatten(x5, start_dim=2)  # Flatten the spatial dimensions (except batch size)
        qcnn_output = self.qcnn(x5_flatten)
        qcnn_output = qcnn_output.view(batch_size, 512, height // 2, width // 2)  # Adjust as per actual output dimensions

        x = self.up1(qcnn_output, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


