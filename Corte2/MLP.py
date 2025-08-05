import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Función para convertir los caracteres a números
def char_convert(mark):
    if mark == '-':
        return 0
    elif mark == 'X':
        return 1
    elif mark == 'O':
        return 2

# Cargar y preparar los datos
size = 10000
board = 3
input_file = []
label_file = []

with open('board_train_data.txt') as fin:
    for _ in range(size):
        tab = []
        fin.readline()
        for _ in range(board):
            line = fin.readline()
            tab.extend([char_convert(line[0]), char_convert(line[2]), char_convert(line[4])])
        fin.readline()
        input_file.append(tab)
        line = fin.readline()
        result = 3 * int(line[0]) + int(line[2])
        output = [1 if i == result else 0 for i in range(board * board)]
        label_file.append(output)

# Convertir a arrays de numpy
X = np.array(input_file)
y = np.array(label_file)

# Crear el modelo
model = Sequential([
    Dense(27, activation='relu', input_shape=(9,)),
    Dense(18, activation='relu'),
    Dense(9, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Guardar el modelo
model.save('tic_tac_toe_model.h5')

# Evaluar el modelo
test_loss, test_accuracy = model.evaluate(X, y)
print(f"Test accuracy: {test_accuracy}")

# Función para hacer una predicción
def predict_move(board):
    board_numeric = [char_convert(cell) for cell in board]
    prediction = model.predict(np.array([board_numeric]))
    return np.argmax(prediction[0])

# Ejemplo de uso
example_board = ['X', 'O', 'X',
                 '-', '-', '-',
                 '-', '-', '-']
best_move = predict_move(example_board)
print(f"Best move for the example board: {best_move}")