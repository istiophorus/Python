def MergeSort(input):

    le = len(input)
    
    if le <= 1:
        return input
    
    if le == 2:
        if (input[0] > input[1]):
            return [input[1], input[0]]
        else:
            return input
        
    inputA = input[0: le // 2]
    inputB = input[le // 2:le]
    
    sortedA = MergeSort(inputA)
    sortedB = MergeSort(inputB)
    
    lenA = len(inputA)
    lenB = len(inputB)
    
    ixA = 0
    ixB = 0
    
    mergedResult = []
    
    while ixA < lenA and ixB < lenB:
        if ixA < lenA and ixB < lenB:
            if sortedA[ixA] <= sortedB[ixB]:
                mergedResult.append(sortedA[ixA])
                ixA += 1
            else:
                mergedResult.append(sortedB[ixB])
                ixB += 1

    mergedResult.extend(sortedA[ixA:lenA])
    mergedResult.extend(sortedB[ixB:lenB])
    
    return mergedResult
   

input_list = [9,8,7,6,5,1,2,3,4]

result = MergeSort(input_list)

print(result)