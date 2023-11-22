import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2


def get_adjacency_matrix(graph_filename, data_filename, normalized_k=0.1):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    adj_df = pd.read_csv(graph_filename, header=None)
    data_df = pd.read_csv(data_filename, header=None)

    num_streets = len(data_df[0])
    dist_mx = np.zeros((num_streets, num_streets), dtype=np.float32)
    dist_mx[:] = np.inf
    
    # Builds street segment to position map
    street_idx_to_pos = {}
    for index, row in data_df.iterrows():
        street_idx_to_pos[index] = (row.loc[2], row.loc[3])
            
    # Loop through all rows and columns
    for index, row in adj_df.iterrows():
        for column, value in row.items():
            dist_mx[index, column] = haversine_distance(street_idx_to_pos[index], street_idx_to_pos[column])
    
    return dist_mx
            
def haversine_distance(point1, point2):
    """
    Calculate the Haversine distance between two points on the Earth.
    
    :param point1: Tuple (latitude1, longitude1)
    :param point2: Tuple (latitude2, longitude2)
    :return: Distance in kilometers
    """
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = radians(point1[0]), radians(point1[1])
    lat2, lon2 = radians(point2[0]), radians(point2[1])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distance in kilometers
    distance = R * c

    return distance