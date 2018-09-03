def BubbleSort(input_list):
    isSorted = False

    offset = 0
    
    while not isSorted:
        isSorted = True
        for ix in range(len(input_list) - 1 - offset):
            if (input_list[ix] > input_list[ix + 1]):
                isSorted = False
                input_list[ix],input_list[ix + 1] = input_list[ix + 1], input_list[ix]
                print(input_list)
        offset += 1


#%%timeit

input_list = [9,8,7,6,5,1,2,3,4]

BubbleSort(input_list)

print(input_list)
            