import numpy as np
import pandas as pd

def read_excel_file(filename, sheet_name): #Función que lee las coordenadas y la demanda de una hoja específica de excel. Las columnas se llaman 'X' e 'Y' y 'Demanda'.
    df = pd.read_excel(filename, sheet_name=sheet_name, header=1) # Lee el data frame y ponemos header=1 para poner la primera fila como índice 
    coordinates = df[['X','Y']].values
    demands = df['Demanda'].values
    return coordinates, demands


def calculate_distance_matrix(coordinates): #Función para calcular la matriz de distancia entre las coordenadas de los sitios a los que hay que llevar los paquetes.
    num_points = len(coordinates)
    dist_matrix = np.zeros((num_points, num_points)) #Creamos una matriz de ceros con el número de filas y de columnas igual al número de coordendas total en el excel.

    for i in range(num_points): #Va recorriendo todas las filas
        for j in range(num_points): #Va recorriendo todas las columnas 
            dist_matrix[i, j] = calculate_distance(coordinates, i, j) #Cada vez que se itera los puntos son distintos y por lo tanto la distancia también cambia creando así la matriz correspondiente. 
    return dist_matrix

def calculate_distance(coordinates, i, j): #Calcula la distancia Euclideana entre dos puntos. Esta función se usará para la segunda parte del código cuando implementemos el algoritmos 2-opt.
    x1, y1 = coordinates[i]
    x2, y2 = coordinates[j]
    return np.sqrt((x1 - x2) **2+ (y1 - y2) **2)


def calculate_total_distance(route, dist_matrix): #Calcular la distancia total de una ruta dada usando la matriz de distancia.
    total_distance = 0
    num_points =len(route)
    for i in range(num_points -1):
        current_node = route[i]
        next_node = route[i +1]
        total_distance += dist_matrix[current_node, next_node]
    return total_distance

def nearest_neighbor(dist_matrix, demands, capacity):#Utiliza el algoritmo del Vecino más Cercano para hallar las rutas que van a realizar los camiones.
    num_points = dist_matrix.shape[0]
    visited = np.zeros(num_points, dtype=bool)
    routes = []

    while np.sum(visited) < num_points:
        current_node = np.where(~visited)[0][0] if any(~visited) else 0 # Start at node 0
        current_capacity = 0
        route = [current_node]
        visited[current_node] = True 

        while True:
            current = route[-1]
            nearest = None
            min_dist = float('inf')

            for neighbor in np.where(~visited)[0]:
                if demands[neighbor] + current_capacity <= capacity:
                    if dist_matrix[current, neighbor] < min_dist:
                        nearest = neighbor
                        min_dist = dist_matrix[current, neighbor]
            if nearest is None:
                break

            route.append(nearest)
            visited[nearest] = True
            current_capacity += demands[nearest]
        routes.append(route)
    return routes



def format_output(routes): #Función que devuelve una lista de las rutas.
    return routes


def vrp_solver(filename, sheet_name, capacity):# Función que resuelve el problema del proyecto haciendo uso de todas las funciones implementadas anteriormente. 
    coordinates, demands = read_excel_file(filename, sheet_name) 
    dist_matrix = calculate_distance_matrix(coordinates)
    routes = nearest_neighbor(dist_matrix, demands, capacity) 
    formatted_routes = format_output(routes)
    return formatted_routes

#Metemos los datos necesarios para el funcionamiento de este código
#La dirección donde se encuentra el excel
filename = r"D:\uni CEU\segundo cuatri\proyecto 1\excel coordendas\ubicaciones exactas península.xlsx" 
print(filename)

sheet_name = "Hoja1" #El nombre de la hoja donde hemos trabajo con el excel
capacity = 50 #La capacidad que pueden tener los camiones
solution = vrp_solver(filename, sheet_name, capacity) #Resolvemos
print(solution)


for route in solution: 
    print(route)



#implementamos la segunda parte del código con el algoritmo 2-opt
def two_opt(routes, dist_matrix, num_iterations): 
    best_routes = routes.copy()

    for _ in range(num_iterations):
        selected_route_idx = np.random.randint(0,len(routes))
        selected_route = routes[selected_route_idx]

        i, j = np.random.randint(1,len(selected_route) -1, size=2)

        if j < i:
            i, j = j, i

        new_route = selected_route.copy()
        new_route[i:j] = selected_route[j -1: i - 1: -1] # Reverse the path b

        new_total_distance = calculate_total_distance(new_route, dist_matrix)
        current_total_distance = calculate_total_distance(best_routes[selected_route_idx], dist_matrix)


        if new_total_distance < current_total_distance:
            best_routes[selected_route_idx] = new_route.copy()
            print('best rouetes', best_routes)

    return best_routes


def vrp_solver2(filename, sheet_name, capacity, num_iterations): #Resolviendo el problema del proyecto usando las funciones anteriores y le nuevo codigo.
    coordinates, demands = read_excel_file(filename, sheet_name)
    dist_matrix = calculate_distance_matrix(coordinates)
    routes = nearest_neighbor(dist_matrix, demands, capacity)

    for i in range(len(routes)):
        route = routes[i]
        optimized_route = two_opt([route], dist_matrix, num_iterations)[0]
        routes[i] = optimized_route

    formatted_routes = format_output(routes)
    return formatted_routes


solucion_final = vrp_solver2(r"D:\uni CEU\segundo cuatri\proyecto 1\excel coordendas\ubicaciones exactas península.xlsx", "Hoja1", 30, 30)
print('Solución final:', solucion_final)