import geopandas as gpd
import matplotlib.pyplot as plt
import pyproj
from shapely.geometry import Point
import TSP
import numpy as np


# turn shp file, which is in long/lat degrees into meters xy axis. treating 0,0 as the depot
def processSHP(filePath, commRadius):
    sensorsD = gpd.read_file(filePath)  # sensors Degrees
    sensorsM = sensorsD  # sensors in meters

    # convert sensorsD to sensorsM, shift numbers close to 0
    def simplifyPlot(gdfInput):
        # Define the input and output coordinate systems
        input_crs = gdfInput.crs
        output_crs = pyproj.CRS.from_epsg(3857)  # EPSG code for WGS 84 / Pseudo-Mercator

        # Perform the coordinate transformation from degrees to meters
        transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)
        gdfInput['geometry'] = gdfInput['geometry'].to_crs(output_crs)
        gdfInput['Easting'], gdfInput['Northing'] = transformer.transform(gdfInput['Easting'].values,
                                                                          gdfInput['Northing'].values)

        # Normalize values
        subX = min(gdfInput['Easting']) - 50
        subY = min(gdfInput['Northing']) - 50
        gdfInput['Easting'] = gdfInput['Easting'] - subX
        gdfInput['Northing'] = gdfInput['Northing'] - subY
        # Define the lambda function to subtract values from a point
        subtract_from_point = lambda point: Point(point.x - subX, point.y - subY)
        # Apply the lambda function to the geometry column
        gdfInput['geometry'] = gdfInput['geometry'].apply(subtract_from_point)

        return gdfInput

    simplifyPlot(sensorsM)
    # Rename the second and third columns
    new_column_names = ['x', 'y']
    sensorsM = sensorsM.rename(columns={sensorsM.columns[1]: new_column_names[0],
                                        sensorsM.columns[2]: new_column_names[1]})
    fig, ax = plt.subplots(figsize=(8, 8))

    # Zoom out plot
    padding = 100
    ax.set_xlim(min(sensorsM['x']) - padding, max(sensorsM['x']) + padding)
    ax.set_ylim(min(sensorsM['y']) - padding, max(sensorsM['y']) + padding)

    # Plot the sensor points
    sensorsM.plot(ax=ax, color='blue', markersize=50, alpha=0.7)

    # Add circles and find hover points
    rangeCircles = sensorsM
    rangeCircles['Communication Range'] = commRadius  # range(40, 40 + len(rangeCircles))
    rangeCircles['geometry'] = sensorsM['geometry'].buffer(rangeCircles['Communication Range'])

    # for circle in rangeCircles['geometry']:
    #     patch = plt.Polygon(circle.exterior.xy, edgecolor='black', facecolor='lime', alpha=0.4)
    #     ax.add_patch(patch)
    # Plot the circles as patches
    for circle in rangeCircles['geometry']:
        x, y = circle.exterior.xy
        vertices = list(zip(x, y))
        patch = plt.Polygon(vertices, edgecolor='black', facecolor='lime', alpha=0.4)
        ax.add_patch(patch)
    # find midpoints of overlapping sections and add them to hoverPoints gdf
    overlapsOf2 = gpd.overlay(df1=rangeCircles, df2=rangeCircles, how='intersection')
    overlapsOf3 = gpd.overlay(df1=overlapsOf2, df2=overlapsOf2, how='intersection')
    hoverPoints = gpd.GeoDataFrame(geometry=overlapsOf3['geometry'].centroid)

    # remove duplicate hover-points by temporarily making a string column
    hoverPoints['geometry_str'] = hoverPoints['geometry'].astype(str)
    hoverPoints = hoverPoints.drop_duplicates(subset='geometry_str')
    hoverPoints = hoverPoints.drop('geometry_str', axis=1)
    hoverPoints.reset_index(drop=True, inplace=True)
    hoverPoints.plot(ax=ax, color='yellow', markersize=10, alpha=1)

    # Set plot title and labels
    ax.set_title('Sensors with Communication Ranges (Meters)')
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    return ax, hoverPoints



# input: gdf with hover points as geometry column
# output: gdf with ordered hover points and new 'distance' column to travel this path
def findMinTravelDistance(hoverPoints):
    points = np.array([(point.x, point.y) for point in hoverPoints.geometry])
    orderedHoverPoints = TSP.hoverPointsTSP(points)
    return orderedHoverPoints


# plot lines between selected geometry points in the order provided
def plotPath(ax, hoverPoints):
    coordinates = np.array([(point.x, point.y) for point in hoverPoints.geometry])
    for i in range(len(coordinates) - 1):
        xValues = [coordinates[i][0], coordinates[i + 1][0]]
        yValues = [coordinates[i][1], coordinates[i + 1][1]]
        ax.plot(xValues, yValues, 'r-')
    ax.plot([coordinates[len(coordinates) - 1][0], 0], [coordinates[len(coordinates) - 1][1], 0], 'r-')  # return path


def testMapping():
    shpFilePath = 'CAF_Sensor_Dataset_2/CAF_sensors.shp'
    communicationRadius = 70
    ax, hoverPoints = processSHP(shpFilePath, communicationRadius)
    # select random subset of 40 rows
    selectedHoverPoints = hoverPoints.sample(40)
    selectedHoverPoints = findMinTravelDistance(selectedHoverPoints)
    plotPath(ax, selectedHoverPoints)
    plt.show()

testMapping()
