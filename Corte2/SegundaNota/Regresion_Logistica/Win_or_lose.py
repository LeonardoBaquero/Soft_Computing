import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Datos de ejemplo
# Datos ficticios que representan estadísticas de fútbol
np.random.seed(42)  # Para reproducibilidad de resultados

# Generacion de 100 partidos ficticios
n_matches = 100 # Juegos simulados 

# Creacion de variables predictoras

data = {
    'posesion': np.random.normal(50, 8, n_matches), # Posesión del balón en porcentaje (%) 50% es la media y el  10% variacion al rededor de la media. 
    'tiros_arco': np.random.randint(1, 12, n_matches), # Tiros al arco. Minimo y maximo de tiros posibles. Genera numeros entre 2 y 14.
    'faltas_cometidas': np.random.randint(8, 18, n_matches), # Faltas cometidas entre 5 y 19 faltas por partido. 
    'corners': np.random.randint(3, 10, n_matches), # Tiros de esquina
    'tarjetas_amarillas': np.random.randint(0, 6, n_matches),
    'pases_completados': np.random.randint(300, 700, n_matches),
    'distancia_recorrida': np.random.normal(110, 10, n_matches)  # en kilómetros
}

# Conversion las variables a un DataFrame de pandas
df = pd.DataFrame(data)

# Creamos la variable objetivo (ganó = 1, perdió = 0)
# La probabilidad de ganar aumenta con mayor posesión y tiros al arco
probabilidad = (df['posesion']/100 * 0.3 + # Divide la posesión entre 100 para convertirla a decimal (50% → 0.5) Multiplica por 0.3, dándole un 30% de peso en la decisión final
                df['tiros_arco']/15 * 0.4 + # Divide los tiros al arco entre 15 para normalizarlos Multiplica por 0.4, dándole un 40% de peso (es el factor más importante)
                df['corners']/12 * 0.3) # Divide los corners entre 12 para normalizarlos Multiplica por 0.3, dándole un 30% de peso

# Media de 0, Desviación estándar de 0.1, Esto simula factores aleatorios que pueden afectar un partid
df['gano'] = (probabilidad + np.random.normal(0, 0.1, n_matches) > 0.5).astype(int) 
# Si es mayor a 0.5, el equipo ganó
# Si es menor a 0.5, el equipo perdió

# Separamos los datos en conjuntos de entrenamiento y prueba
X = df[['posesion', 'tiros_arco', 'faltas_cometidas', 'corners']]  # Variables predictoras
y = df['gano']  # Variable objetivo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Se crea y entrenamos el modelo de regresión logística
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Realización de predicciones con el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluación del modelo
print("\nPrecisión del modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Función para predecir el resultado de un nuevo partido
def predecir_partido(posesion, tiros_arco, faltas, corners):
    """
    Realiza una predicción sobre el resultado de un partido.

    Parámetros:
    - posesion (float): Porcentaje de posesión del balón.
    - tiros_arco (int): Número de tiros al arco.
    - faltas (int): Número de faltas cometidas.
    - corners (int): Número de tiros de esquina.

    Retorna:
    - prediccion (int): 1 si se espera que el equipo gane, 0 si se espera que pierda.
    - probabilidad (array): Probabilidad de ganar y perder.
    """
    nuevo_partido = np.array([[posesion, tiros_arco, faltas, corners]])
    prediccion = modelo.predict(nuevo_partido)
    probabilidad = modelo.predict_proba(nuevo_partido)
    
    return prediccion[0], probabilidad[0]

# Ejemplo de uso de la función predecir_partido
print("\n--- Predicción para un nuevo partido ---")
pos = 60  # Posesión del 60%
tiros = 10  # 10 tiros al arco
faltas = 12  # 12 faltas
corners = 8  # 8 tiros de esquina

resultado, prob = predecir_partido(pos, tiros, faltas, corners)
print(f"\nEstadísticas del partido:")
print(f"Posesión: {pos}%")
print(f"Tiros al arco: {tiros}")
print(f"Faltas cometidas: {faltas}")
print(f"Corners: {corners}")
print(f"\nPredicción: {'Ganará' if resultado == 1 else 'Perderá'}")
print(f"Probabilidad de ganar: {prob[1]*100:.2f}%")
print(f"Probabilidad de perder: {prob[0]*100:.2f}%")

# Importancia de las variables en el modelo
coef = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': modelo.coef_[0]
})
print("\nImportancia de las variables:")
print(coef.sort_values('Coeficiente', ascending=False))
