import numpy as np
import random

class Triquis: # Definicion de la clase Triqui y sus reglas 
    def __init__(self):
        #  # Inicializa el tablero como una lista de 9 espacios vacíos ('-') y establece el jugador actual como 'X'
        self.board = ['-' for _ in range(9)]
        self.current_player = 'X'

    def make_move(self, position): # Realiza un movimiento en la posición indicada si está disponible
        if self.board[position] == '-': # Verifica si la posición está vacía
            self.board[position] = self.current_player # Coloca la marca del jugador actual en el tablero
            self.current_player = 'O' if self.current_player == 'X' else 'X'  # Cambia al siguiente jugador
            return True # Movimiento válido
        return False  # Movimiento inválido si la posición ya está ocupada

    def check_winner(self): # Comprueba si hay un ganador o empate
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Horizontales
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Verticales
            [0, 4, 8], [2, 4, 6]  # Diagonales
            # Tablero del triqui ejemplificado. Representacion como lista de 9 elementos 
            #  0 | 1 | 2
            #  ---------
            #  3 | 4 | 5
            #  ---------
            #  6 | 7 | 8
                    
        ]
        for combo in winning_combinations: 
            # Comprueba si alguna de las combinaciones ganadoras tiene el mismo símbolo y no está vacía
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != '-':
                return self.board[combo[0]] # Retorna el ganador ('X' o 'O')
        if '-' not in self.board: # Si no hay espacios vacíos y no hay ganador, es empate
            return 'Tie' # Si no hay ganador ni empate, el juego sigue
        return None # Si no hay ganador ni empate, el juego sigue
        
    def get_state(self): # Retorna el estado actual del tablero como una cadena de texto
        return ''.join(self.board)

class QLearningAgent: # Se implementa esta clase que utiliza Q-learning para aprender a jugar (Algoritmo de Q-learning)
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.q_table = {} # Tabla Q que almacena el valor de estado-acción: Es un diccionario que almacena los valores Q para cada par estado-acción.
        self.epsilon = epsilon  # Tasa de exploración (probabilidad de tomar una acción aleatoria) 
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento que valora las recompensas futuras
        
    def get_action(self, state, available_actions): # Selecciona una acción usando la política epsilon-greedy
        if random.uniform(0, 1) < self.epsilon: # Explora aleatoriamente con probabilidad epsilon
            return random.choice(available_actions)
        else:
            # Explota el conocimiento de la tabla Q, eligiendo la acción con el mayor valor Q
            q_values = [self.q_table.get((state, action), 0) for action in available_actions]
            max_q = max(q_values) 
            best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
            return random.choice(best_actions) # Si hay varias mejores acciones, elige una al azar
            
    # Actualiza la tabla Q con la nueva información de la experiencia (estado, acción, recompensa, siguiente estado)
    def update_q_table(self, state, action, reward, next_state): 
        current_q = self.q_table.get((state, action), 0) # Obtiene el valor Q actual para el estado-acción
        # Calcula el valor Q máximo para el siguiente estado
        next_max_q = max([self.q_table.get((next_state, a), 0) for a in range(9)]) 
        # Actualiza el valor Q utilizando la fórmula de actualización Q-learning
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q) 
        self.q_table[(state, action)] = new_q

class QLearningAgent: # Se implementa esta clase que utiliza Q-learning para aprender a jugar
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.q_table = {}
        self.epsilon = epsilon  # Exploración vs explotación
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento

        """ """

    def get_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        else:
            q_values = [self.q_table.get((state, action), 0) for action in available_actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
            return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table.get((state, action), 0)
        next_max_q = max([self.q_table.get((next_state, a), 0) for a in range(9)])
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[(state, action)] = new_q

# Función para cargar los datos de entrenamiento del tablero desde un archivo
def load_training_data(filename):
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 5):
            # Check if enough lines are left to avoid IndexError
            if i + 4 < len(lines):  
                board = ''.join(lines[i:i+3]).replace('\n', '').replace(' ', '')
                # Handle potential errors in move formatting
                try:
                    move_parts = lines[i+4].split()
                    if len(move_parts) >= 2: # Check if the line has at least 2 elements after splitting
                        move = int(move_parts[0]) * 3 + int(move_parts[1])
                        data.append((board, move))
                    else:
                        print(f"Warning: Invalid move format in line {i+5}. Skipping.")
                except ValueError:
                    print(f"Warning: Invalid move format in line {i+5}. Skipping.")
            else:
                print(f"Warning: Incomplete data at line {i+1}. Skipping.")
    return data

def initialize_q_table(agent, filename): # Inicializa la tabla Q con datos de entrenamiento
    training_data = load_training_data(filename)
    for board, move in training_data:
        state = board
        action = move
        agent.q_table[(state, action)] = 1 # Inicializa el valor Q de esos estados-acciones        

def train_agent(episodes=10000): # Función para entrenar al agente jugando múltiples episodios contra sí mismo
    agent = QLearningAgent() # Crea el agente
    initialize_q_table(agent, 'board_train_data.txt') # Carga los datos de entrenamiento
    initialize_q_table(agent, 'board_data.txt') # Carga más datos si están disponibles
    for _ in range(episodes):
        game = Triquis()
        state = game.get_state() # Obtiene el estado inicial del tablero
        while True:
            available_actions = [i for i, mark in enumerate(game.board) if mark == '-'] # Acciones posibles
            action = agent.get_action(state, available_actions) # Selecciona una acción
            game.make_move(action) # Realiza el movimiento
            next_state = game.get_state() # Obtiene el siguiente estado
            winner = game.check_winner() # Comprueba si hay ganador
            if winner: # Si hay un ganador o empate, asigna una recompensa
                reward = 1 if winner == 'X' else -1 if winner == 'O' else 0
                agent.update_q_table(state, action, reward, next_state) # Actualiza la tabla Q
                break
            agent.update_q_table(state, action, 0, next_state) # Actualiza la tabla Q sin recompensa
            state = next_state # Actualiza el estado actual
    return agent

# Función para jugar contra el agente entrenado
def play_game(agent): 
    game = Triquis()
    while True:
        print(f"\nCurrent board: {''.join(game.board)}") # Tablero actual 
        if game.current_player == 'X': # El agente juega como 'X'
            available_actions = [i for i, mark in enumerate(game.board) if mark == '-']
            action = agent.get_action(game.get_state(), available_actions)
        else: # El usuario juega como 'O'
            action = int(input("Enter your move (0-8): ")) # Solicita un movimiento al usuario
        if game.make_move(action): # Realiza el movimiento
            winner = game.check_winner() # Comprueba si hay un ganador o empate
            if winner:
                print(f"\nFinal board: {''.join(game.board)}")
                if winner == 'Tie':
                    print("It's a tie!") # Empate!
                else:
                    print(f"{winner} wins!") # Gana!
                break
        else:
            print("Invalid move, try again.") # Movimineto incorrecto, vuelva a intentar 

# Entrenar al agente
trained_agent = train_agent()

# Jugar contra el agente entrenado
play_game(trained_agent)

def initialize_q_table(agent, filename): 
    training_data = load_training_data(filename)
    for board, move in training_data:
        state = board
        action = move
        agent.q_table[(state, action)] = 1        

def train_agent(episodes=10000): # Define una función train_agent que entrena al agente jugando múltiples partidas contra sí mismo.
    agent = QLearningAgent()
    initialize_q_table(agent, 'board_train_data.txt')
    initialize_q_table(agent, 'board_data.txt')
    for _ in range(episodes):
        game = Triquis()
        state = game.get_state()
        while True:
            available_actions = [i for i, mark in enumerate(game.board) if mark == '-']
            action = agent.get_action(state, available_actions)
            game.make_move(action)
            next_state = game.get_state()
            winner = game.check_winner()
            if winner:
                reward = 1 if winner == 'X' else -1 if winner == 'O' else 0
                agent.update_q_table(state, action, reward, next_state)
                break
            agent.update_q_table(state, action, 0, next_state)
            state = next_state
    return agent

def play_game(agent): # Incluye una función play_game que permite jugar contra el agente entrenado.
    game = Triquis()
    while True:
        print(f"\nCurrent board: {''.join(game.board)}")
        if game.current_player == 'X':
            available_actions = [i for i, mark in enumerate(game.board) if mark == '-']
            action = agent.get_action(game.get_state(), available_actions)
        else:
            action = int(input("Enter your move (0-8): "))
        if game.make_move(action):
            winner = game.check_winner()
            if winner:
                print(f"\nFinal board: {''.join(game.board)}")
                if winner == 'Tie':
                    print("It's a tie!")
                else:
                    print(f"{winner} wins!")
                break
        else:
            print("Invalid move, try again.")

# Entrenar al agente
trained_agent = train_agent()

# Jugar contra el agente entrenado
play_game(trained_agent)