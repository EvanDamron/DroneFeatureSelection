import geopandas as gpd
import matplotlib.pyplot as plt
import pyproj
from shapely.geometry import Point
import TSP
import numpy as np
from SetCoverPy import setcover
from ML import processData
import random


# turn shp file, which is in long/lat degrees into meters xy axis. treating 0,0 as the depot
def processSHP(filePath, height, commRadius):
    sensors = gpd.read_file(filePath)

    # convert degrees to meters and shift numbers close to 0
    def simplifyPlot(gdfInput):
        # Define the input and output coordinate systems
        input_crs = gdfInput.crs
        output_crs = pyproj.CRS.from_epsg(3857)  # EPSG code for WGS 84 Pseudo-Mercator

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

    sensors = simplifyPlot(sensors)
    # Rename the second and third columns
    new_column_names = ['x', 'y']
    sensors = sensors.rename(columns={sensors.columns[1]: new_column_names[0],
                                      sensors.columns[2]: new_column_names[1]})
    fig, ax = plt.subplots(figsize=(8, 8))

    # Zoom out plot
    padding = 100
    ax.set_xlim(min(sensors['x']) - padding, max(sensors['x']) + padding)
    ax.set_ylim(min(sensors['y']) - padding, max(sensors['y']) + padding)

    # Plot the sensor points
    sensors.plot(ax=ax, color='blue', markersize=25, alpha=0.7)

    hoverPoints, sensorNames = getHoverPoints(sensors, commRadius, height, ax)
    # Set plot title and labels
    ax.set_title('Sensors with Communication Ranges (Meters)')
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    return fig, ax, hoverPoints, sensorNames


def getHoverPoints(sensors, commRadius, height, ax):
    # Add circles and find hover points
    droneRadius = (commRadius ** 2 - height ** 2) ** 0.5
    rangeCircles = sensors.copy()
    rangeCircles['Communication Range'] = droneRadius
    rangeCircles['geometry'] = sensors['geometry'].buffer(rangeCircles['Communication Range'])
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

    # create dictionary to correspond hover points to sensors
    sensorNames = {}
    for hoverPoint in hoverPoints['geometry']:
        sensorNames[hoverPoint] = []
        for circle in rangeCircles['geometry']:
            if hoverPoint.within(circle):
                sensorName = rangeCircles.loc[rangeCircles['geometry'] == circle, 'Location'].values[0]
                sensorNames[hoverPoint].append(sensorName)
    return hoverPoints, sensorNames


def generateSensorsUniformRandom(height, commRadius, df, numSensors, areaLength=1000, areaWidth=1000,
                                 minDistance=80):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-100, areaWidth + 100)
    ax.set_ylim(-100, areaLength + 100)
    if numSensors < len(df.columns):
        columnsToKeep = np.random.choice(df.columns, size=numSensors, replace=False)
        df = df[columnsToKeep]
    elif numSensors > len(df.columns):
        for i in range(numSensors - len(df.columns)):
            oldColumnName = df.columns[i]
            newColumnName = f"SYNTHETIC{i + 1}"
            df[newColumnName] = df[oldColumnName] * 0.95
    sensorPoints = []
    for _ in range(numSensors):
        while True:
            x = random.uniform(0, areaLength)
            y = random.uniform(0, areaWidth)
            newPoint = Point(x, y)

            if all(newPoint.distance(existingPoint) >= minDistance for existingPoint in sensorPoints):
                sensorPoints.append(Point(x, y))
                break
    sensorsGDF = gpd.GeoDataFrame(geometry=sensorPoints)
    sensorsGDF['Location'] = df.columns
    sensorsGDF.plot(ax=ax, markersize=30, color='blue')
    hoverPoints, sensorNames = getHoverPoints(sensorsGDF, commRadius, height, ax)
    return fig, ax, hoverPoints, sensorNames, df


# input: gdf with hover points as geometry column
# output: gdf with ordered hover points and new 'distance' column to travel this path
def findMinTravelDistance(hoverPoints, scramble=False):
    points = np.array([(point.x, point.y) for point in hoverPoints.geometry])
    if not scramble:
        orderedHoverPoints, distance = TSP.hoverPointsTSP(points)
    else:
        orderedHoverPoints, distance = TSP.hoverPointsTSP(points, scramble=True)
    return orderedHoverPoints, distance


# plot lines between selected geometry points in the order provided
def plotPath(ax, hoverPoints):
    coordinates = np.array([(point.x, point.y) for point in hoverPoints.geometry])
    ax.plot([coordinates[0][0], 0], [coordinates[0][1], 0], 'r-')
    for i in range(len(coordinates) - 1):
        xValues = [coordinates[i][0], coordinates[i + 1][0]]
        yValues = [coordinates[i][1], coordinates[i + 1][1]]
        ax.plot(xValues, yValues, 'r-')
    ax.plot([coordinates[len(coordinates) - 1][0], 0], [coordinates[len(coordinates) - 1][1], 0], 'r-')  # return path


# Only keeps the smallest amount of hoverpoints needed to cover all the sensors, used in RSEO
def minSetCover(sensorNames, hoverPoints):
    allSensors = sorted(set(sensor for sensors in sensorNames.values() for sensor in sensors))
    # Create a mapping of sensors to their indices in the matrix
    sensorIndexes = {sensor: i for i, sensor in enumerate(allSensors)}
    numRows = len(allSensors)
    numCols = len(sensorNames)
    aMatrix = np.zeros((numRows, numCols), dtype=int)
    for j, (point, sensors) in enumerate(sensorNames.items()):
        for sensor in sensors:
            i = sensorIndexes[sensor]
            aMatrix[i, j] = True
    sums = np.sum(aMatrix, axis=0)
    cost = [value - ((value - 1) * .1) for value in sums]
    solver = setcover.SetCover(amatrix=aMatrix, cost=cost)
    result = solver.SolveSCP()
    decisionList = solver.s
    filteredHoverPoints = hoverPoints[decisionList]
    filteredHoverPoints.reset_index(drop=True, inplace=True)
    validPoints = set(filteredHoverPoints['geometry'])
    filteredSensorNames = {point: value for point, value in sensorNames.items() if point in validPoints}
    return filteredSensorNames, filteredHoverPoints


def getSensorNames(points, sensorNames):
    corrSensorsSet = set()
    for point in points:
        if point in sensorNames:
            corrSensorsSet.update(sensorNames[point])
    return list(corrSensorsSet)


def testMapping():
    shpFilePath = 'CAF_Sensor_Dataset_2/CAF_sensors.shp'
    communicationRadius = 70
    height = 15
    # fig, ax, hoverPoints, sensorNames = processSHP(shpFilePath, height, communicationRadius)
    dataFolder = 'CAF_Sensor_Dataset_2/caf_sensors/Hourly'
    df = processData(dataFolder)
    rSeed = 1
    random.seed(rSeed)
    np.random.seed(rSeed)
    # pathPlotName = f"Experiment Maps/Sensors60Seed{rSeed}"
    fig, ax, hoverPoints, sensorNames, df = generateSensorsUniformRandom(height, communicationRadius, df,
                                                        numSensors=len(df.columns), areaLength=1000, areaWidth=1000)
    hoverPoints.plot(ax=ax, color='yellow', markersize=25)
    for key, value in sensorNames.items():
        if len(value) == 1:
            ax.text(key.x, key.y, str(value[0]), fontsize=10, ha='center', va='bottom')
    # fig.savefig(pathPlotName, bbox_inches='tight')
    plt.show()

# testMapping()
