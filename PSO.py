import numpy as np

# Parámetros de PSO
num_particulas = 30
dimensiones = 2
iteraciones = 100
c1 = 2.0  # Coeficiente cognitivo
c2 = 2.0  # Coeficiente social
w = 0.7   # Inercia o peso 

# Límites de búsqueda de las particulas
limite_inferior = -5.12
limite_superior = 5.12

# Inicialización de las partículas
# Se inicializan las mejores posiciones y valores personales de cada partícula.
# Se inicializa la mejor posición y valor global del enjambre.
posiciones = np.random.uniform(limite_inferior, limite_superior, (num_particulas, dimensiones))
velocidades = np.random.uniform(-1, 1, (num_particulas, dimensiones))
mejores_posiciones_personales = np.copy(posiciones)
# La función de aptitud utilizada en este caso es la función de Rastrigin, definida 
mejores_valores_personales = np.apply_along_axis(lambda x: 10*dimensiones + sum(x**2 - 10*np.cos(2*np.pi*x)), 1, posiciones)
mejor_valor_global = np.min(mejores_valores_personales)
mejor_posicion_global = posiciones[np.argmin(mejores_valores_personales)]

# Función de aptitud
def funcion_rastrigin(x):
    return 10*dimensiones + sum(x**2 - 10*np.cos(2*np.pi*x))

# Iteraciones de PSO
for iteracion in range(iteraciones):
    for i in range(num_particulas):
        valor_personal_actual = funcion_rastrigin(posiciones[i])
        
        # Actualización de la mejor posición personal
        if valor_personal_actual < mejores_valores_personales[i]:
            mejores_valores_personales[i] = valor_personal_actual
            mejores_posiciones_personales[i] = posiciones[i]

        # Actualización de la mejor posición global
        if valor_personal_actual < mejor_valor_global:
            mejor_valor_global = valor_personal_actual
            mejor_posicion_global = posiciones[i]

    # Actualización de la velocidad y la posición
    for i in range(num_particulas):
        velocidades[i] = (w * velocidades[i] +
                          c1 * np.random.rand() * (mejores_posiciones_personales[i] - posiciones[i]) +
                          c2 * np.random.rand() * (mejor_posicion_global - posiciones[i]))
        posiciones[i] = posiciones[i] + velocidades[i]
        posiciones[i] = np.clip(posiciones[i], limite_inferior, limite_superior)

    print(f"Iteración {iteracion+1}/{iteraciones}, Mejor valor global: {mejor_valor_global}")

# Resultados
print("Mejor posición encontrada:", mejor_posicion_global)
print("Mejor valor encontrado:", mejor_valor_global)

# Al final del bucle de optimización, se imprimen la mejor posición encontrada y el mejor valor encontrado.

