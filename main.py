from script import torchInput, torchOutput, testInput, testOutput
from Model import LinearRegression
import torch

print(torchInput)
print(torchOutput)

inputSize = 18
outputSize = 1

model = LinearRegression(inputSize, outputSize) # input and output size are 1
print(model)


criterion = torch.nn.MSELoss(size_average = False) 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) 

for epoch in range(500): 
  
    # Forward pass: Compute predicted y by passing  
    # x to the model 
    pred_y = model(torchInput) 
  
    # Compute and print loss 
    loss = criterion(pred_y, torchOutput) 
  
    # Zero gradients, perform a backward pass,  
    # and update the weights. 
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 
    print('epoch {}, loss {}'.format(epoch, loss.data[0])) 
