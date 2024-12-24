from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np
import pandas as pd
from geopy.distance import distance
import requests
from tqdm import tqdm

# Function to extract courier id and date
def get_id_kurir_and_date(df: pd.DataFrame):
    df['id_kurir'] = df['pod__photo'].str.extract(r'camera\.photoDeliveryProcessImage\.(\d+)')
    df['pod__timereceive'] = pd.to_datetime(df['pod__timereceive'])
    df['date'] = df['pod__timereceive'].dt.date
    df['date'] = df['date'].astype(str)
    df['id_kurir'] = df['id_kurir'].astype(str)
    return df

# Function to split latitude and longitude
def get_long_lat(df: pd.DataFrame):
    df[['latitude', 'longitude']] = df['pod__coordinate'].str.split(',', expand=True)
    # Convert the data type of both columns to float
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    return df

def get_osrm_distance_matrix(coordinates):
    base_url = "http://router.project-osrm.org/table/v1/driving/"
    coord_string = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
    response = requests.get(f"{base_url}{coord_string}?annotations=distance")
    
    if response.status_code == 200:
        data = response.json()
        # Matriks jarak dalam meter
        return np.array(data['distances'])
    else:
        print("Failed to fetch distance matrix:", response.text)
        return None

# Function to create a distance matrix using OSRM
def create_distance_matrix_with_osrm(df):
    # Extract coordinates as a list of tuples (latitude, longitude)
    coordinates = df[['latitude', 'longitude']].apply(tuple, axis=1).tolist()
    # Get the OSRM distance matrix
    osrm_matrix = get_osrm_distance_matrix(coordinates)
    
    if osrm_matrix is not None:
        connote_codes = df['connote__connote_code'].values  # Extract connote codes for later use
        distance_matrix = []
        for i in range(len(osrm_matrix)):
            distance_row = []
            for j in range(len(osrm_matrix[i])):
                distance_row.append({
                    'distance': osrm_matrix[i][j],  # Use OSRM distance
                    'connote_code_i': connote_codes[i],
                    'connote_code_j': connote_codes[j]
                })
            distance_matrix.append(distance_row)
        return distance_matrix
    else:
        raise ValueError("Failed to create distance matrix with OSRM.")

# Function to create a distance map with kantor coordinates added using OSRM
def create_distance_map_with_kantor(df, df_kantor):
    # Extract the kantor's latitude and longitude
    kantor_lat = df_kantor['latitude'].iloc[0]
    kantor_long = df_kantor['longitude'].iloc[0]
    nama_kantor = df_kantor['nama_kantor'].iloc[0]

    couriers = df['id_kurir'].unique()
    dates = df['date'].unique()
    distance_map = []
    
    for courier in couriers:
        for date in dates:
            # Filter data for the specific courier and date
            courier_df = df[(df['id_kurir'] == courier) & (df['date'] == date)].copy()
            
            # Add kantor coordinates as the first "row" in the filtered DataFrame
            kantor_data = pd.DataFrame([{
                'latitude': kantor_lat,
                'longitude': kantor_long,
                'connote__connote_code': nama_kantor
            }])

            extended_df = pd.concat([kantor_data, courier_df], ignore_index=True)
            
            # Create the distance matrix including the kantor coordinates
            distance_matrix = create_distance_matrix_with_osrm(extended_df)
            
            # Append to the distance map
            distance_map.append([courier, date, distance_matrix])
    
    all_distance_map = pd.DataFrame(distance_map, columns=['id_kurir', 'tanggal', 'distance_matrix_per_kurir'])
    return all_distance_map

# Function to print the solution which also prints the connote code
def print_solution(manager, routing, solution, distance_matrix):
    index = routing.Start(0)
    route = []
    route_distance = 0
    connote_codes = []  # List to store connote codes for the route
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route.append(node_index)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        
        # Add the connote code associated with this node
        connote_codes.append(distance_matrix[previous_index][node_index]['connote_code_i'])
        
    # Add the final node
    node_index = manager.IndexToNode(index)
    route.append(node_index)
    connote_codes.append(distance_matrix[previous_index][node_index]['connote_code_j'])
    
    return route, route_distance, connote_codes

# Update solve_tsp function to include connote codes in the output
def solve_tsp(all_distance_map):
    num_cities = len(all_distance_map)

    # Create data manager
    manager = pywrapcp.RoutingIndexManager(num_cities, 1, 0)  # 1 vehicle, starting from node 0

    # Create routing model
    routing = pywrapcp.RoutingModel(manager)

    # Travel cost between nodes
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(all_distance_map[from_node][to_node]['distance'])  # Must be integer

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Define search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 30  # Time limit 30 seconds

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Display the solution
    if solution:
        return print_solution(manager, routing, solution, all_distance_map)
    else:
        print("No solution found.")
        return None, None, None

# Update fuel consumption and price conversion inside the function
def get_optimal_route_with_codes(df, km_per_liter_fuel=40, fuel_price_per_liter=13000):
    all_route = []
    
    for i in range(len(df)):
        # Solve TSP for the optimal route
        optimal_route_indices, optimal_distance, optimal_connote_codes = solve_tsp(df.loc[i, 'distance_matrix_per_kurir'])
        
        # Get original route and calculate total distance
        original_route_indices = list(range(len(df.loc[i, 'distance_matrix_per_kurir'])))  # Original order of indices
        original_connote_codes = [
            df.loc[i, 'distance_matrix_per_kurir'][index][index]['connote_code_i'] 
            for index in original_route_indices
        ]  # Get connote codes for original route
        
        # Calculate original distance including return to start
        original_distance = 0
        for j in range(len(original_route_indices) - 1):
            original_distance += df.loc[i, 'distance_matrix_per_kurir'][original_route_indices[j]][original_route_indices[j + 1]]['distance']
        
        # Add distance from last node back to the starting node
        original_distance += df.loc[i, 'distance_matrix_per_kurir'][original_route_indices[-1]][original_route_indices[0]]['distance']
        
        # Round the original distance
        original_distance = round(original_distance)
        
        # Calculate the difference in distance and efficiency
        distance_difference = round(original_distance - optimal_distance)
        route_efficiency = (distance_difference / original_distance) * 100 if original_distance > 0 else 0
        
        # Round and format values
        optimal_distance = round(optimal_distance)
        route_efficiency = f"{round(route_efficiency, 2)}%"
        
        # Calculate fuel savings and cost savings
        fuel_savings = round((distance_difference / 1000) / km_per_liter_fuel, 2)  # in liters
        cost_savings = round(fuel_savings * fuel_price_per_liter)  # Convert cost based on price per liter

        # Append all calculated metrics
        all_route.append([ 
            df.loc[i, 'id_kurir'], 
            df.loc[i, 'tanggal'], 
            original_connote_codes, 
            original_distance, 
            optimal_connote_codes, 
            optimal_distance, 
            distance_difference, 
            route_efficiency, 
            fuel_savings, 
            cost_savings
        ])
    
    # Create DataFrame with all metrics
    all_optimal_route = pd.DataFrame(all_route, columns=[ 
        'courier_id', 
        'date', 
        'route_original', 
        'total_distance_original (meters)', 
        'route_optimal', 
        'total_distance_optimal (meters)', 
        'distance_difference (meters)', 
        'route_efficiency', 
        'fuel_savings (liter)', 
        'cost_savings (rupiah)'
    ])
    return all_optimal_route

# Main function
def predict_optimal_route(df: pd.DataFrame, df_kantor, km_per_liter_fuel = 40, fuel_price_per_liter=13000) -> pd.DataFrame:
    df = get_id_kurir_and_date(df)
    df = get_long_lat(df)
    result_distance_map = create_distance_map_with_kantor(df,df_kantor)
    rute_optimal = get_optimal_route_with_codes(result_distance_map, km_per_liter_fuel, fuel_price_per_liter)
    return rute_optimal