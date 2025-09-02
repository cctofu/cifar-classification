# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

class BatchNorm1d(nn.Module):
    # TODO START
    def __init__(self, num_features, momentum=1e-2, eps=1e-5):
        super(BatchNorm1d, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(num_features, device=device))
        self.bias = nn.Parameter(torch.zeros(num_features, device=device))

        self.register_buffer('running_mean', torch.zeros(num_features, device=device))
        self.register_buffer('running_var', torch.ones(num_features, device=device))

    def forward(self, input):
        if not self.training:
            BN = (input - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            return self.weight * BN + self.bias
        
        input_mean = input.mean([0])
        input_var = input.var([0], unbiased=False)
        
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * input_mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * input_var
        
        BN = (input - input_mean) / torch.sqrt(input_var + self.eps)
        res = self.weight * BN + self.bias

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
		return input * torch.bernoulli((1-self.p)*torch.ones_like(input,device=input.device))
	# TODO END

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=10, drop_rate=0.5):
        super(Model, self).__init__()
        # TODO START
        # Define your layers here
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.bn1 = nn.BatchNorm1d(hidden_dim)       
        self.relu = nn.ReLU()                       
        self.dropout = nn.Dropout(drop_rate)        
        self.fc2 = nn.Linear(hidden_dim, output_dim) 
        # TODO END

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # TODO START
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        # TODO END

        pred = torch.argmax(logits, 1)  # Calculate the prediction result
        if y is None:
            return pred
        loss = self.loss(logits, y)
        correct_pred = (pred.int() == y.int())
        acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

        return loss, acc

