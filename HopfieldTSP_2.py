import numpy as np
import matplotlib.pyplot as plt

def create_clean_tsp_visualization(coordinates, path):
    # Crear figura con fondo blanco
    plt.figure(figsize=(10, 10), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Remover ejes y ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Dibujar las conexiones entre ciudades
    for i in range(len(coordinates)):
        start = coordinates[i]
        end = coordinates[path[i]]
        plt.plot([start[0], end[0]], 
                [start[1], end[1]], 
                'k-',  # línea negra
                linewidth=1)  # grosor de línea
    
    # Dibujar los puntos de las ciudades
    plt.plot(coordinates[:,0], 
            coordinates[:,1], 
            'r.',  # puntos rojos
            markersize=8)  # tamaño de los puntos
    
    # Ajustar los límites y márgenes
    plt.margins(0.1)
    
    # Mantener la proporción del aspecto
    plt.axis('equal')
    
    return plt

# Datos de las ciudades del Sahara Occidental. Coordenadas 
coordinates = np.array([
    [20833.3333, 17100.0000], [28900.0000, 17066.6667],
    [21300.0000, 13016.6667], [21600.0000, 14150.0000],
    [21600.0000, 14966.6667], [21600.0000, 16500.0000],
    [22183.3333, 13133.3333], [22583.3333, 14300.0000],
    [22683.3333, 12716.6667], [23616.6667, 15866.6667],
    [23700.0000, 15933.3333], [23883.3333, 14533.3333],
    [24166.6667, 13250.0000], [25149.1667, 12365.8333],
    [26133.3333, 14500.0000], [26150.0000, 10550.0000],
    [26283.3333, 12766.6667], [26433.3333, 13433.3333],
    [26550.0000, 13850.0000], [26733.3333, 11683.3333],
    [27026.1111, 13051.9444], [27096.1111, 13415.8333],
    [27153.6111, 13203.3333], [27166.6667, 9833.3333],
    [27233.3333, 10450.0000], [27233.3333, 11783.3333],
    [27266.6667, 10383.3333], [27433.3333, 12400.0000],
    [27462.5000, 12992.2222]
])

# Ejemplo de ruta (esto normalmente vendría de tu algoritmo TSP)
# Aquí se usó una ruta de ejemplo
path = np.arange(len(coordinates))
np.random.shuffle(path)

# Crear la visualización
plt = create_clean_tsp_visualization(coordinates, path)
plt.show()