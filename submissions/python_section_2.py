import pandas as pd
import numpy as np
import datetime

def calculate_distance_matrix(file_path):
    df = pd.read_csv(file_path)

    #to extract unique toll location ids
    unique_ids = sorted(set(df['id_start']).union(set(df['id_end'])))
    
    #to map toll IDs to matrix indices
    id_to_index = {id_: idx for idx, id_ in enumerate(unique_ids)}
    
    n = len(unique_ids)
    distance_matrix = np.full((n, n), np.inf)
    
    np.fill_diagonal(distance_matrix, 0)

    #to populate the matrix with the given distances from the data
    for _, row in df.iterrows():
        start_idx = id_to_index[row['id_start']]
        end_idx = id_to_index[row['id_end']]
        distance_matrix[start_idx, end_idx] = row['distance']
        distance_matrix[end_idx, start_idx] = row['distance']  

    #Floyd-Warshall Algorithm to compute shortest paths between pairs
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[k, j]:
                    distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]

    distance_df = pd.DataFrame(distance_matrix, index=unique_ids, columns=unique_ids)
    
    return distance_df

def unroll_distance_matrix(distance_df):
    unrolled_data = []

    #to iterate over the unique IDs
    for id_start in distance_df.index:
        for id_end in distance_df.columns:
            if id_start != id_end:
                distance = distance_df.loc[id_start, id_end]
                #to append the result to the list as a tuple
                unrolled_data.append((id_start, id_end, distance))

    #to convert the list of tuples to a DataFrame
    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
    
    #to keep only relevant id_start and id_end pair
    filtered_unrolled_df = unrolled_df[unrolled_df['distance'] != np.inf]

    return filtered_unrolled_df.reset_index(drop=True)


def find_ids_within_ten_percentage_threshold(unrolled_df, reference_id):
    #to filter the DataFrame
    distances = unrolled_df[unrolled_df['id_start'] == reference_id]['distance']
    
    if distances.empty:
        return []
    
    #to calculate the average distance for the reference ID
    average_distance = distances.mean()
    
    #to calculate the lower and upper bounds (10% threshold)
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1
    
    #to filter the DataFrame for IDs within the 10% threshold
    result_ids = unrolled_df[
        (unrolled_df['distance'] >= lower_bound) & 
        (unrolled_df['distance'] <= upper_bound)
    ]['id_start']
    
    #to get unique IDs and sort them
    unique_sorted_ids = sorted(result_ids.unique())
    
    return unique_sorted_ids

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    #to define the rate for each vehicle type
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    #to calculate toll rates for each vehicle
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate
    
    return df

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    #days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    #time intervals for weekdays and corresponding factors
    time_intervals = [
        (datetime.time(0, 0), datetime.time(10, 0), 0.8),  # 00:00 to 10:00
        (datetime.time(10, 0), datetime.time(18, 0), 1.2),  # 10:00 to 18:00
        (datetime.time(18, 0), datetime.time(23, 59, 59), 0.8)  # 18:00 to 23:59
    ]
    
    #constant factor for weekends
    weekend_factor = 0.7
    
    rows = []
    
    #to iterate through all unique (id_start, id_end) pairs and days of the week
    for _, row in df.iterrows():
        for day in days_of_week:
            #entries for each time interval
            for start_time, end_time, factor in time_intervals:
                #to calculate adjusted toll rates
                adjusted_moto = row['moto'] * factor
                adjusted_car = row['car'] * factor
                adjusted_rv = row['rv'] * factor
                adjusted_bus = row['bus'] * factor
                adjusted_truck = row['truck'] * factor

                #to append the new row to the list
                rows.append({
                    'id_start': row['id_start'],
                    'id_end': row['id_end'],
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    'moto': adjusted_moto,
                    'car': adjusted_car,
                    'rv': adjusted_rv,
                    'bus': adjusted_bus,
                    'truck': adjusted_truck
                    
                })

            #add entries for weekends with the constant discount factor
            if day in ['Saturday', 'Sunday']:
                adjusted_moto = row['moto'] * weekend_factor
                adjusted_car = row['car'] * weekend_factor
                adjusted_rv = row['rv'] * weekend_factor
                adjusted_bus = row['bus'] * weekend_factor
                adjusted_truck = row['truck'] * weekend_factor

                rows.append({
                    'id_start': row['id_start'],
                    'id_end': row['id_end'],
                    'start_day': day,
                    'start_time': datetime.time(0, 0),  
                    'end_day': day,
                    'end_time': datetime.time(23, 59, 59),
                    'moto': adjusted_moto,
                    'car': adjusted_car,
                    'rv': adjusted_rv,
                    'bus': adjusted_bus,
                    'truck': adjusted_truck
                      
                })

    time_based_df = pd.DataFrame(rows)
    
    return time_based_df


file_path = 'dataset-2.csv'
result_df = calculate_distance_matrix(file_path)
unrolled_df = unroll_distance_matrix(result_df)
toll_rate_df = calculate_toll_rate(unrolled_df)
time_based_toll_df = calculate_time_based_toll_rates(toll_rate_df)


print(result_df)
print(unrolled_df)
print(toll_rate_df)
print(time_based_toll_df)