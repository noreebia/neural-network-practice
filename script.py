import torch;

file = open ("data.txt", "r")

for line in file:
    lineToStringArray = line.split(",")
    print(line, end='')
    break

file.close()