import numpy as np
#from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.heuristics import solve_tsp_local_search
import geopandas as gpd
from shapely import Point



def hoverPointsTSP(points):
    depotLocation = (0, 0)
    points = np.insert(points, 0, depotLocation, axis=0)
    distanceMatrix = euclidean_distance_matrix(points)
    permutation, distance = solve_tsp_local_search(distanceMatrix)
    orderedPoints = [points[i] for i in permutation]
    gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in orderedPoints], crs='EPSG:3857')
    gdf['distance'] = distance
    return gdf
