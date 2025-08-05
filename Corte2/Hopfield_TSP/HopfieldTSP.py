import numpy as np
import matplotlib.pyplot as plt

class SaharaTSPSolver:
    def __init__(self):
        # Coordenadas de las 29 ciudades del Sahara Occidental. 
        self.coordinates = np.array([
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
        
        self.n_cities = len(self.coordinates)
        self.distances = self._calculate_distances()
        
    def _calculate_distances(self):
        """Calcula la matriz de distancias entre todas las ciudades"""
        distances = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                distances[i,j] = np.sqrt(np.sum((self.coordinates[i] - self.coordinates[j])**2))
        return distances
    
    def solve(self, max_iter=2000, T=0.02, A=1.5, B=1.5, C=2.0, D=1.0):
        """Resuelve el TSP usando una red de Hopfield"""
        # Inicialización de la matriz de estados
        V = np.random.random((self.n_cities, self.n_cities))
        energies = []
        
        for iteration in range(max_iter):
            dV = np.zeros_like(V)
            
            # Actualización paralela de todas las neuronas
            for i in range(self.n_cities):
                for j in range(self.n_cities):
                    # Cálculo del delta para cada neurona
                    delta = 0
                    
                    # Restricción de una ciudad por paso
                    delta -= A * (2 * np.sum(V[i,:]) - 1)
                    
                    # Restricción de un paso por ciudad
                    delta -= B * (2 * np.sum(V[:,j]) - 1)
                    
                    # Minimización de distancia
                    for k in range(self.n_cities):
                        if k != (j+1) % self.n_cities:
                            delta -= C * self.distances[i,k] * V[k,(j+1)%self.n_cities]
                        if k != (j-1) % self.n_cities:
                            delta -= C * self.distances[i,k] * V[k,(j-1)%self.n_cities]
                    
                    # Restricción del número total de pasos
                    delta -= D * (2 * np.sum(V) - 2*self.n_cities)
                    
                    # Actualización usando función sigmoide
                    dV[i,j] = 1 / (1 + np.exp(-delta/T)) - V[i,j]
            
            V += dV
            
            # Cálculo de la energía actual
            energy = self._calculate_energy(V, A, B, C, D)
            energies.append(energy)
            
            # Criterio de convergencia
            if iteration > 100 and np.abs(energies[-1] - energies[-2]) < 0.0001:
                break
        
        return V, energies
    
    def _calculate_energy(self, V, A, B, C, D):
        """Calcula la energía total de la red"""
        E1 = sum((np.sum(V[i,:]) - 1)**2 for i in range(self.n_cities))
        E2 = sum((np.sum(V[:,j]) - 1)**2 for j in range(self.n_cities))
        
        E3 = 0
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                for k in range(self.n_cities):
                    if k != (j+1) % self.n_cities:
                        E3 += self.distances[i,k] * V[i,j] * V[k,(j+1)%self.n_cities]
        
        E4 = (np.sum(V) - self.n_cities)**2
        
        return A*E1 + B*E2 + C*E3 + D*E4
    
    def visualize_solution(self, solution, energies):
        """Visualiza la solución y la evolución de la energía"""
        path = np.argmax(solution, axis=1)
        
        plt.figure(figsize=(15, 6))
        
        # Graficar ruta
        plt.subplot(121)
        plt.scatter(self.coordinates[:,0], self.coordinates[:,1], c='blue', label='Ciudades')
        
        # Dibujar conexiones
        for i in range(self.n_cities):
            start = self.coordinates[i]
            end = self.coordinates[path[i]]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'r-', alpha=0.5)
        
        # Marcar primera ciudad
        plt.scatter(self.coordinates[0,0], self.coordinates[0,1], c='green', s=100, label='Inicio')
        
        plt.title('Ruta TSP en Sahara Occidental')
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.legend()
        
        # Graficar energía
        plt.subplot(122)
        plt.plot(energies)
        plt.title('Evolución de la Energía')
        plt.xlabel('Iteraciones')
        plt.ylabel('Energía')
        
        plt.tight_layout()
        plt.show()
        
        # Calcular distancia total
        total_distance = 0
        for i in range(self.n_cities):
            total_distance += self.distances[i, path[i]]
        
        return total_distance

# Ejemplo de uso
solver = SaharaTSPSolver()

# Resolver con parámetros optimizados para problema grande
solution, energies = solver.solve(
    max_iter=2000,
    T=0.02,  # Temperatura baja para convergencia más estable
    A=1.5,   # Mayor peso en restricciones de ciudad única
    B=1.5,   # Mayor peso en restricciones de paso único
    C=2.0,   # Énfasis en minimización de distancia
    D=1.0    # Peso moderado en restricción de pasos totales
)

# Visualizar y obtener distancia total
total_distance = solver.visualize_solution(solution, energies)
print(f"Distancia total del recorrido: {total_distance:.2f} unidades")