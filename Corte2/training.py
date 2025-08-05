import random
from rich import print

size  = 10000
board = 3
psize = int(size * 4 / 5)

index = range(size)
index_select = random.sample(index, psize)
other_index  = list(set(index) - set(index_select))

input_file = []
label_file = []

def char_convert(mark):
    if mark == '-':
        return 0
    elif mark == 'X':
        return 1
    elif mark == 'O':
        return 2

with open('board_train_data.txt') as fin:
    for _ in range(size):
        tab = []
        fin.readline()
        for _ in range(board):
            line = fin.readline()
            tab.append(char_convert(line[0]))
            tab.append(char_convert(line[2]))
            tab.append(char_convert(line[4]))
        fin.readline()
        input_file.append(tab)
        line = fin.readline()
        result = 3 * int(line[0]) + int(line[2])
        output = []
        for i in range(board * board):
            if i == result:
                output.append(1)
            else:
                output.append(0)
        label_file.append(output)

# Datos de Entrada
inputs = [input_file[i] for i in index_select]

# Rotulos
labels = [label_file[i] for i in index_select]

print(list(zip(inputs, labels)))
