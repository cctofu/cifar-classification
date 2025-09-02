# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

class BatchNorm2d(nn.Module):
	# TODO START
    def __init__(self, num_features, momentum=1e-2, eps=1e-5):
        super(BatchNorm2d, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(num_features).to(device))
        self.bias = nn.Parameter(torch.zeros(num_features).to(device))

        self.register_buffer('running_mean', torch.zeros(num_features).to(device))
        self.register_buffer('running_var', torch.ones(num_features).to(device))
        
    def forward(self, input): 
        if not self.training:
            BN = (input - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.eps)
            return BN * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        input_mean = input.mean([0, 2, 3])
        input_var = input.var([0, 2, 3], unbiased=False)
        
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * input_mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * input_var
        
        BN = (input - input_mean.view(1, -1, 1, 1)) / torch.sqrt(input_var.view(1, -1, 1, 1) + self.eps)
        res = BN * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        
        return res
	# TODO END

class Dropout(nn.Module):
    # TODO START
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, input): 
        if self.training is False: 
            return input 
        return input * torch.bernoulli((1-self.p)*torch.ones_like(input, device=input.device))
    # TODO END

class Model(nn.Module):
    def __init__(self, num_classes=10, drop_rate=0.5):
        super(Model, self).__init__()
		# TODO START
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = Dropout(p=drop_rate)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = Dropout(p=drop_rate)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Final Linear layer
        self.fc = nn.Linear(128 * 8 * 8, num_classes)

		# TODO END
        # Loss
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
		# TODO START
        # Pass input through the layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.maxpool2(x)

        # Flatten the tensor for the linear layer
        x = x.view(x.size(0), -1)
        logits = self.fc(x)

        pred = torch.argmax(logits, 1)
        if y is None:
            return pred

        loss_val = self.loss(logits, y)
        correct_pred = (pred.int() == y.int())
        acc = torch.mean(correct_pred.float())

        return loss_val, acc
		# TODO END

