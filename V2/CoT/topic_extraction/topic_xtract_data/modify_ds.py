# THIS CELL IS JUST USED FOR DATASET MANIPULATION
file = open('./CoT/generated_ds.csv', 'r')
f2 = open('./CoT/ds.txt', 'w')
lines = file.readlines()

for line in lines:
    idx = line.find(':')
    l2 = '[' + line[:idx]  + ']|' + line[idx+1:]
    f2.write(l2)