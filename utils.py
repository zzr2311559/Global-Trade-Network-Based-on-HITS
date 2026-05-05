import os
from geopy.distance import geodesic

# Trade distance (GCD)
def calculate_trade_distance(coord1:tuple, coord2:tuple) -> float:
    return geodesic(coord1, coord2).km # Great Circle Distance