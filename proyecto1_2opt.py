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
    visited = np.zeros(num_points, dtype=bool) #Lleva registro de de los puntos que se van visitando
    routes = [] #Lista vacía para ir añadiendo las rutas

    while np.sum(visited) < num_points: #Genera rutas hasta que todos los puntos han sido visitados
        current_node = np.where(~visited)[0][0] if any(~visited) else 0 #Selecciona el primer punto no visitado como el nodo actual para empezar una nueva ruta. Si todos los puntos están visitados empieza en el punto 0.
        current_capacity = 0
        route = [current_node]
        visited[current_node] = True #Marca el nodo actual como visitado

        while True:
            current = route[-1] #Se va a buscar el vecino más cercano al último punto añadido a la ruta
            nearest = None
            min_dist = float('inf') #Inicializamos la distancia mínima en infinito para poder encontrar la menor distancia posible

            for neighbor in np.where(~visited)[0]: #Este bucle recorre todos los puntos no visitados
                if demands[neighbor] + current_capacity <= capacity: #Si agregar este vecino no excede la capacidad del vehículo y la distancia es menor que la distancia mínima actual, actualiza nearest y min_dist
                    if dist_matrix[current, neighbor] < min_dist:
                        nearest = neighbor
                        min_dist = dist_matrix[current, neighbor]
            if nearest is None: #Si no encuentra ningún vecino apto a estas características rompe el bucle para finalizar la ruta.
                break

            route.append(nearest) #añade el vecino más cercano
            visited[nearest] = True
            current_capacity += demands[nearest] #Actualiza la capacidad con la demanda del vecino añadido
        routes.append(route) #Cuando ya no hay más vecinos que recorrer se devuelve la lista de rutas.
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
print("La capacidad de las furgonetas es:", capacity)


for i, route in enumerate(solution, start=1): 
        print('Ruta y furgoneta número', i,':', route)



#implementamos la segunda parte del código con el algoritmo 2-opt
def two_opt(routes, dist_matrix, num_iterations): 
    best_routes = routes.copy() #Hacemos una copia de las rutas para no modificar las originales y para poder comprobar si se han hecho mejoras

    for _ in range(num_iterations):
        selected_route_idx = np.random.randint(0,len(routes)) #Índice aleatorio para seleccionar una ruta específica de la lista de rutas
        selected_route = routes[selected_route_idx] #Esta va a ser la ruta elegida según el índice aleatorio

        if len(selected_route) < 3: #Si la ruta tiene menos de 3 nodos, no se puede aplicar el algoritmo 2-opt
            continue

        i, j = np.random.randint(1,len(selected_route) -1, size=2) #Desginar una valor para i y j dentro de las rutas seleccionadas para intentar encontrar una mejora mediante la reversión del orden de visita entre estos dos puntos

        if j < i: #Este if se asegura que i sea siempre menor que j 
            i, j = j, i

        new_route = selected_route.copy() #Copia de las rutas seleccionadas. 
        new_route[i:j] = selected_route[j -1: i - 1: -1] #Invierte la ruta para ver si la distancia es menor de la nueva manera

        new_total_distance = calculate_total_distance(new_route, dist_matrix) #Calcula distancia total de la nueva ruta modificada
        current_total_distance = calculate_total_distance(best_routes[selected_route_idx], dist_matrix) #Calcula la distancia total de la ruta actual antes de la modificación


        if new_total_distance < current_total_distance: #Si la distancia total de la nueva ruta es menor que la distancia total de la ruta actual, actualiza la ruta en best_routes con la nueva ruta. Esto provoca una nueva versión más eficiente de las rutas.
            best_routes[selected_route_idx] = new_route.copy()
            print('best routes', best_routes) #Se imprime cuando se halla una nueva ruta.

    return best_routes


def vrp_solver2(filename, sheet_name, capacity, num_iterations): #Resolviendo el problema del proyecto usando las funciones anteriores y le nuevo codigo.
    coordinates, demands = read_excel_file(filename, sheet_name)
    dist_matrix = calculate_distance_matrix(coordinates) #Calcula y devuelve una matriz de distancias entre todos los puntos basada en sus coordenadas.
    routes = nearest_neighbor(dist_matrix, demands, capacity)

    for i in range(len(routes)): #Recorre todas las rutas generadas por la función del vecino más cercano
        route = routes[i]
        optimized_route = two_opt([route], dist_matrix, num_iterations)[0] #Se le aplica a cada ruta para intentar mejorarla. 
        routes[i] = optimized_route

    formatted_routes = format_output(routes)
    return formatted_routes #Devuelve las rutas 


solucion_final = vrp_solver2(r"D:\uni CEU\segundo cuatri\proyecto 1\excel coordendas\ubicaciones exactas península.xlsx", "Hoja1", 50, 50)
for i, route in enumerate(solucion_final, start=1): #Imprime la solución final
        print('Ruta solución final y furgoneta número', i,':', route)