a = [[]] * 2
a # [[], []] -> You think, you got 2D array
# You want to append to first bracket one element
a[0].append(1)
print(a) # [[1], [1]] -> It was added to both brackets for no reason
