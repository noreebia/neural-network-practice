from script import trainingInput, trainingOutput, testInput, testOutput
from model import LinearRegression
import torch

print(trainingInput)
print(trainingOutput)

print(trainingInput.size())
print(trainingOutput.size())

inputSize = 18
outputSize = 1

model = LinearRegression(inputSize, outputSize)
print(model)

criterion = torch.nn.MSELoss(size_average = False) 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) 

for epoch in range(500): 
  
    # Forward pass: Compute predicted y by passing  
    # x to the model 
    predictedOutput = model(trainingInput) 
  
    # Compute and print loss 
    loss = criterion(predictedOutput, trainingOutput) 
  
    # Zero gradients, perform a backward pass,  
    # and update the weights. 
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 
    print('epoch {}, loss {}'.format(epoch, loss.data[0])) 
