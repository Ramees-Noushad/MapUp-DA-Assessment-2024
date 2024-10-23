from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    start = 0 #start is 0, the variable tracks the begining of current group that will reverrsed
   
    while start < len(lst):  # while loop continues as long as start is less than the length of the list
        end = min(start + n, len(lst))  # end is set to smaller of start+n or length of the list
        
        left = start #left starts at the begining of the current group
        right = end -1 # right is the last index of the current group
        while left < right: # swap the elements from left and right end until they cross each other.
            lst[left], lst[right] = lst[right],lst[left] 
            left += 1
            right -= 1
        start = end # start is moved to the next group of n elements
        
    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    dictionary = {}
    
    for i in lst:
        length = len(i) #for each i calculates its length and stores in length variable

        if length not in dictionary:
            dictionary[length] = [] #if the length is not present as a key in the dictionary, it adds a new key-value pair
        dictionary[length].append(i)  #append the i to the list corresponding to its length

    sorted_dict = dict(sorted(dictionary.items())) 

    return sorted_dict

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flattened = {}

    def flatten(d, prefix=""): #to perform the recursive flattering
        for k, v in d.items(): #iterate through each key value pair 
            if isinstance(v, dict): 
                flatten(v, f"{prefix}{k}{sep}") #if the value is dictionary, the new prefix is constructed by appending the current key(k) to the existing prefix
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    flatten({k + f"[{i}]": item}, prefix)  # if the value is a list, it iterate through each element of the list. The index is used to reference the each element.
            else:
                flattened[prefix + k] = v #if the value is neither a dictionary nor a list, the key is concatinating the prefix and the key. and the value is assigned to the key in the dictionary

    flatten(nested_dict)
    return flattened
    return dict

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    result = []
    used = [False] * len(nums) #to create a list named used as the same lengthof input nums. Each elemant is set to false in the begining to note that the elements form nums have not used yet.

    def back(path):
        if len(path) == len(nums):
            result.append(list(path))
            return  #if the length of the current permutation equals the lengths of nums, then permutation completes. the permutation adds to result.
        
        for i in range(len(nums)): # loop iterate over the elements in the num by therir indices
            if not used[i]: # to check the elemnt is used or not, not used then added to permutation
                used[i] = True #mark the element as used
                path.append(nums[i]) #add the element to permutation
                back(path) #to continue the current permutation for updated path
                path.pop() #removes the last element added to path
                used[i] = False #mark the current one as unused
    back([])
    return result

import re
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Define regular expressions for the three supported date formats
    date_pattern = r'\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\b'
    
    # Find all matches of the date pattern in the text
    matches = re.findall(date_pattern, text)
    
    return matches

import polyline 
from geopy.distance import geodesic  #library calculates he distance between 2 points using haversine formula
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str) #Decode the polyline string into a list of latitude and longitude
    
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude']) #converts the decoded coordiates into Pandas dataframe with columns for latitude and longitude
    
    distances = [0]  #first point made as 0 to calculate the distance
    for i in range(1, len(coordinates)): #loop iterate from 2nd elemant to last element
        prev_point = coordinates[i - 1] 
        current_point = coordinates[i]
        dist = geodesic(prev_point, current_point).meters # Use geopy's geodesic method to calculate the distance between two points
        distances.append(dist)
    
    df['distance'] = distances
    
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    rotated = [[0] * n for p in range(n)]

   
    for i in range(n):
        for j in range(n):
            rotated[j][n - i - 1] = matrix[i][j]  # To rotate the matrix 90 degrees clockwise
      
    transform = [[0] * n for p in range(n)] #initialise the transformed matrix
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated[i])  # Sum of all elements in the current row
            col_sum = sum(rotated[k][j] for k in range(n))  # Sum of all elements in the current column
            transform[i][j] = row_sum + col_sum - 2 * rotated[i][j]  # Exclude the current element from both sums

    return transform


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    df['startDay'] = pd.to_numeric(df['startDay'], errors='coerce').fillna(0).astype(int) # Convert startDay and endDay to integers if they are not already
    df['endDay'] = pd.to_numeric(df['endDay'], errors='coerce').fillna(0).astype(int)

    
    def check_complete_time(group): # A function to check completeness for each group of (id, id_2)
        
        full_days = set(range(7))  # Assuming 0 = Monday, 6 = Sunday
        covered_days = set()
        covered_times = set()

        for _, row in group.iterrows(): # Convert the start and end time to datetime objects
            
            start_day = row['startDay']
            end_day = row['endDay']
            start_time = pd.to_datetime(row['startTime'], format='%H:%M:%S').time()
            end_time = pd.to_datetime(row['endTime'], format='%H:%M:%S').time()

           
            covered_days.update(range(start_day, end_day + 1)) # Record covered days

            
            if start_day == end_day:  # Generate the time range covered by this entry
                time_range = pd.date_range(
                    start=pd.Timestamp.combine(pd.to_datetime('today'), start_time),
                    end=pd.Timestamp.combine(pd.to_datetime('today'), end_time),
                    freq='s' 
                ).time
            else: 
                
                end_of_day_time = pd.Timestamp.combine(pd.to_datetime('today'), pd.to_datetime('23:59:59').time()) # Add time from start_day to the end of that day
                time_range_start = pd.date_range(
                    start=pd.Timestamp.combine(pd.to_datetime('today'), start_time),
                    end=end_of_day_time,
                    freq='s' 
                ).time
                
               
                time_range_end = pd.date_range(
                    start=pd.Timestamp.combine(pd.to_datetime('today') + pd.Timedelta(days=1), pd.to_datetime('00:00:00').time()),
                    end=pd.Timestamp.combine(pd.to_datetime('today') + pd.Timedelta(days=1), end_time),
                    freq='s'  
                ).time   # Add time from the start of end_day to end_time
                
                time_range = list(time_range_start) + list(time_range_end)

            
            covered_times.update(time_range) # Update covered times

        
        complete_days = len(full_days) == len(covered_days) # Check if all days are covered and if the time range spans 24 hours
        complete_time = len(covered_times) == (24 * 60 * 60)  # 86400 seconds in 24 hours

        return complete_days and complete_time

   
    result = df.groupby(['id', 'id_2']).apply(check_complete_time)  # Apply the completeness check to each (id, id_2) group
    
    return ~result 
