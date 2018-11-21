import torch      
from torch.autograd import Variable     
import torch.nn as nn 
import warnings
warnings.filterwarnings("ignore")

# create class
class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(LinearRegression,self).__init__()
        # Linear function.
        self.linear = nn.Linear(input_size,output_size)

    def forward(self,x):
        return self.linear(x)

inputSize = 18
outputSize = 4

model = LinearRegression(inputSize, outputSize) # input and output size are 1
