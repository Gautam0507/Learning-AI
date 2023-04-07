from array import * 
array1 = array("i", [10,22,33,78,16,90,23,66,87,90])
sum = 0
for num in array1:
    sum += num
mean = sum/len(array1)
array1.insert(len(array1),int(sum/len(array1)))
for num in array1:
    print(num)