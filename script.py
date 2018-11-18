import torch;

file = open ("data.txt", "r")

outputList = []
intListOfLists = []
for line in file:
    if line.startswith("@") or line.startswith("\n"):
        continue
    
    lineToStringArray = line.split(",")
    print(lineToStringArray)
    intList = []
    for index, string in enumerate(lineToStringArray):
        if index == 2 or index ==3 or index == 4 or index == 5:
            continue
        elif index == 11 or index == 16 or index == 17 or index == 20 or index == 21:
            if string == "Large" or string == "Good" or string == "Good\n":
                intList.append(0)
            elif string == "Average" or string == "Average\n":
                intList.append(1)
            elif string == "Small" or string == "Poor" or string == "Poor\n":
                intList.append(2) 
            else:
                print("Unidentified string!!! At index" + str(index) + ":" + string)
        else:
            if string == "M" or string == "G" or string == "Y" or string == "Married" or string == "T" or string == "Free" or string == "Vh" or string == "Large" or string == "Il" or string == "Service" or string == "Govt" or string == "Eng":
                intList.append(0)
            elif string == "F" or string == "ST" or string == "N" or string == "Unmarried" or string == "V" or string == "Paid" or string == "High" or string == "Average" or string == "Um" or string == "Business" or string == "Private" or string == "Asm":
                intList.append(1)
            elif string == "SC" or string == "Am" or string == "10" or string == "Retired" or string == "Hin":
                intList.append(2)
            elif string == "OBC" or string == "Medium" or string == "12" or string == "Farmer" or string == "Housewife" or string == "Ben":
                intList.append(3)
            elif string == "MOBC" or string == "Low" or string == "Degree" or string == "Others" or string == "Ben":
                intList.append(4)
            elif string == "Pg":
                intList.append(5)
            else:
                print("Unidentified string!!! At index" + str(index) + ":" + string)
    print(intList)
    print()
    intListOfLists.append(intList)

print(intListOfLists)
file.close()