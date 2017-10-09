

import torch
from torch import autograd, nn
import torch.nn.functional as F


batch_size = 10
input_size = 4
hidden_size = 5
num_classes = 3

torch.manual_seed(1234)

input  = autograd.Variable(torch.rand(batch_size, input_size))
print("\nInput: ", input)


class Model(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, num_classes)
        self.h3 = nn.Softmax()

    def forward(self, x):
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        x = self.h3(x)
        return x

model = Model(input_size=input_size,
              hidden_size=hidden_size, 
              num_classes=num_classes)


##### Added for backprop
learning_rate = 0.001

target = autograd.Variable((torch.rand(batch_size) * num_classes).long())
print("\nTarget: ", target)

from torch import optim
opt = optim.SGD(params=model.parameters(), lr=learning_rate)

output_probs = model(input)
print('out', output_probs)

























