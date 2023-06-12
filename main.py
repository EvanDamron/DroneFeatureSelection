#epsilon squared greedy algorithm to find the optimal set of hover points to predict the other points without
#violating distance budget (simplification of energy-of-drone budget

from mapping import processSHP, findMinTravelDistance, plotPath, getSensorNames
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from ML import getMSE, processData
import random

positionFilePath = 'CAF_Sensor_Dataset_2/CAF_sensors.shp'
dataFolderPath = 'CAF_Sensor_Dataset_2/caf_sensors/Hourly'
communicationRadius = 70
maxDistance = 4070
# HP: Hover Points S:selected U:unselected
ax, HP_gdf, sensorNames = processSHP(positionFilePath, communicationRadius)
df = processData(dataFolderPath)

SHP_gdf = gpd.GeoDataFrame()
SHP_gdf['geometry'] = None
SHP_gdf = SHP_gdf.set_geometry('geometry')
UHP_gdf = HP_gdf.copy()  #To test and drastically reduce runtime use #HP_gdf.head(20)
SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
UHP_gdf.crs = 'EPSG:3857'
outOfBudget = False

#unused for now, may need to do this in every iteration of while loop to compute arProb
def getPointsInBudget(unselected, selected):
    inBudgetHP = unselected.copy()
    for index, row in unselected.iterrows():
        extractedRow = gpd.GeoDataFrame([row], geometry='geometry', crs=unselected.crs)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, extractedRow], ignore_index=True), crs=unselected.crs)
        tempSHP = findMinTravelDistance(tempSHP)
        if tempSHP['distance'][0] < maxDistance:
            inBudgetHP = inBudgetHP.drop(index)
        else:
            inBudgetHP[index]['distanceIfAdded'] = tempSHP['distance'][0]
# Adds a random hoverpoint that is within budget         #RESET INDEX???
def addRandomHP(unselected, selected):
    print('random add')
    tempUHP = unselected.copy()
    while not tempUHP.empty:
        randomRow = tempUHP.sample(n=1)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, randomRow], ignore_index=True), crs=selected.crs)
        tempSHP = findMinTravelDistance(tempSHP)
        tempUHP = tempUHP.drop(index=randomRow.index)
        if tempSHP['distance'][0] <= maxDistance:
            unselected = unselected.drop(index=randomRow.index)
            return unselected, tempSHP
    global outOfBudget
    outOfBudget = True
    return unselected, selected


def remRandomHP(unselected, selected):
    print('random remove')
    if selected.empty:
        return unselected, selected
    randomRow = selected.sample(n=1)
    unselected = gpd.GeoDataFrame(pd.concat([unselected, randomRow], ignore_index=True), crs=unselected.crs)
    selected = selected.drop(index=randomRow.index)
    return unselected, selected

def addBestHP(unselected, selected):
    print("add best")
    global distanceOfBest
    rewards = [-1] * len(unselected)
    if selected.empty:   #never happens in actual greedy alg
        oldDistance = 0
        oldMSE = float('inf')
    else:
        oldDistance = selected['distance'][0]
        features = getSensorNames(selected['geometry'], sensorNames)
        oldMSE = getMSE(features, df)
    for index, row in unselected.iterrows():
        print('index ', index, 'of', len(unselected))
        extractedRow = gpd.GeoDataFrame([row], geometry='geometry', crs=unselected.crs)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, extractedRow], ignore_index=True), crs=unselected.crs)
        tempSHP = findMinTravelDistance(tempSHP)
        if tempSHP['distance'][0] <= maxDistance:
            features = getSensorNames(tempSHP['geometry'], sensorNames)
            newMSE = getMSE(features, df)
            # print(tempSHP, features, newMSE, 'x')
            rewards[index] = (oldDistance - tempSHP['distance'][0]) * (oldMSE - newMSE)
            if rewards[index] == max(rewards):
                distanceOfBest = tempSHP['distance'][0]
    maxReward = max(rewards)
    maxIndex = rewards.index(maxReward)
    rowToMove = unselected.loc[[maxIndex]].copy()
    rowToMove.loc[:, 'distance'] = distanceOfBest
    selected = gpd.GeoDataFrame(pd.concat([selected, rowToMove], ignore_index=True), crs=unselected.crs)
    unselected = unselected.drop(maxIndex)
    return unselected, selected


def remBestHP(unselected, selected):
    print('REMOVE BEST')
    global distanceOfBest
    rewards = [-1] * len(selected)
    if selected.empty:   #never happens in actual greedy alg
        return unselected, selected
    elif len(selected) == 1:
        return remRandomHP(unselected, selected)
    else:
        oldDistance = selected['distance'][0]
        features = getSensorNames(selected['geometry'], sensorNames)
        oldMSE = getMSE(features, df)
    for index, row in selected.iterrows():
        print('index ', index, 'of', len(selected))
        tempSHP = selected.drop(index=index)
        tempSHP = findMinTravelDistance(tempSHP)
        if tempSHP['distance'][0] <= maxDistance:
            features = getSensorNames(tempSHP['geometry'], sensorNames)
            newMSE = getMSE(features, df)
            # print(tempSHP, features, newMSE, 'x')
            rewards[index] = (oldDistance - tempSHP['distance'][0]) * (oldMSE - newMSE)
            if rewards[index] == max(rewards):
                distanceOfBest = tempSHP['distance'][0]
    maxReward = max(rewards)
    maxIndex = rewards.index(maxReward)
    rowToMove = selected.loc[[maxIndex]].copy()
    rowToMove.loc[:, 'distance'] = distanceOfBest
    unselected = gpd.GeoDataFrame(pd.concat([unselected, rowToMove], ignore_index=True), crs=selected.crs)
    selected = selected.drop(maxIndex)
    return unselected, selected

arProb = 0 #probability of adding (0) and removing (1)
rbProb = 1 #probability of random (1) and best (0)
L = 20 # number of loops
loopCount = 0
while rbProb > 0:
    loopCount += 1
    print("Loop iteration ", loopCount)
    raProb = rbProb * (1 - arProb) #random add
    rrProb = rbProb * arProb #random remove
    baProb = (1 - rbProb) * (1 - arProb) #best add
    brProb = (1 - rbProb) * arProb # best remove
    randomNumber = (random.random())
    if randomNumber < raProb:
        UHP_gdf, SHP_gdf = addRandomHP(UHP_gdf, SHP_gdf)
    elif randomNumber < raProb + rrProb:
        UHP_gdf, SHP_gdf = remRandomHP(UHP_gdf, SHP_gdf)
    elif randomNumber < raProb + rrProb + baProb:
        UHP_gdf, SHP_gdf = addBestHP(UHP_gdf, SHP_gdf)
    else:
        UHP_gdf, SHP_gdf = remBestHP(UHP_gdf, SHP_gdf)
    if len(SHP_gdf) == 0 | len(UHP_gdf) == 0:
        break
    UHP_gdf = UHP_gdf.reset_index(drop=True)
    SHP_gdf = SHP_gdf.reset_index(drop=True)
    rbProb = rbProb - 1/L
    features = getSensorNames(SHP_gdf['geometry'], sensorNames)
    mse = getMSE(features, df)
    if SHP_gdf.empty:
        totalDistance = 0
    else:
        totalDistance = SHP_gdf['distance'][0]

    print('Total Distance Traveled: ', totalDistance)
    print('Total Number of Hover Points Visited: ', len(SHP_gdf))
    print('mse: ', mse)
    arProb = totalDistance / maxDistance #as we approach budget, more likely to remove

plotPath(ax, SHP_gdf)
plt.show()
