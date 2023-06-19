# epsilon squared greedy algorithm to find the optimal set of hover points to predict the other features without
# violating distance budget (simplification of energy-of-drone budget)

from mapping import processSHP, findMinTravelDistance, plotPath, getSensorNames
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from ML import getMSE, processData
import random
from joblib import Parallel, delayed
import time
startTime = time.time()
positionFilePath = 'CAF_Sensor_Dataset_2/CAF_sensors.shp'
dataFolderPath = 'CAF_Sensor_Dataset_2/caf_sensors/Hourly'
communicationRadius = 70
maxDistance = 4000

# HP: Hover Points S:selected U:unselected
ax, HP_gdf, sensorNames = processSHP(positionFilePath, communicationRadius)
# HP_gdf = HP_gdf.sample(n=20)  # Use this to test, reduces runtime
df = processData(dataFolderPath)

SHP_gdf = gpd.GeoDataFrame()
SHP_gdf['geometry'] = None
SHP_gdf = SHP_gdf.set_geometry('geometry')
UHP_gdf = HP_gdf.copy()
SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
UHP_gdf.crs = 'EPSG:3857'
outOfBudget = False


# determine what points can be visited without violating distance budget
def getPointsInBudget(unselected, selected):
    inBudgetHP = unselected.copy()
    def calculateDistanceIfAdded(index, row):
        extractedRow = gpd.GeoDataFrame([row], geometry='geometry', crs=unselected.crs)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, extractedRow], ignore_index=True), crs=unselected.crs)
        tempSHP = findMinTravelDistance(tempSHP)
        distanceIfAdded = tempSHP['distance'][0]
        return index, distanceIfAdded

    indexesDistances = Parallel(n_jobs=-1)(delayed(calculateDistanceIfAdded)(index, row) for index, row in inBudgetHP.iterrows())
    indexesDistances.sort(key=lambda x: x[0])
    for value in indexesDistances:
        index, distance = value
        if distance > maxDistance:
            inBudgetHP = inBudgetHP.drop(index)
        else:
            inBudgetHP.at[index, 'distanceIfAdded'] = distance
    if 'distance' in inBudgetHP.columns:
        inBudgetHP.drop('distance', axis=1, inplace=True)
    inBudgetHP = inBudgetHP.reset_index(drop=True)
    return inBudgetHP

# Adds a random hoverpoint that is in budget (IB)
def addRandomHP(unselected, unselectedIB, selected):
    print('ADD RANDOM')
    if unselectedIB.empty:
        print('No more hover points within budget')
        return unselected, selected
    randomRow = unselectedIB.sample(n=1)
    randomRow.rename(columns={'distanceIfAdded': 'distance'}, inplace=True)
    randomRow = randomRow.reset_index(drop=True)
    pointAdded = randomRow.iloc[0]['geometry']
    unselected = unselected[unselected['geometry'] != pointAdded].reset_index(drop=True)
    SHP = gpd.GeoDataFrame(pd.concat([selected, randomRow], ignore_index=True), crs=selected.crs)
    SHP['distance'] = randomRow['distance'][0]
    return unselected, SHP


# Remove a random hover point from selected
def remRandomHP(unselected, selected):
    print('REMOVE RANDOM')
    if selected.empty:
        return selected
    rowToMove = selected.sample(n=1)
    UHP = gpd.GeoDataFrame(pd.concat([unselected, rowToMove], ignore_index=True), crs=unselected.crs)
    if 'distance' in UHP.columns:
        UHP.drop('distance', axis=1, inplace=True)
    selected = selected.drop(index=rowToMove.index)
    return UHP, selected


# Add the best in-budget-hoverpoint to selected, best = max((oldDistance - newDistance) * (oldMSE - newMSE))

def addBestHP(unselected, unselectedIB, selected):
    print("ADD BEST")
    if unselectedIB.empty:
        print('No more hover points within budget')
        return unselected, selected
    rewards = [float('-inf')] * len(unselectedIB)
    if selected.empty:  # never happens in actual greedy alg
        oldDistance = 0
        oldMSE = 999999999
    else:
        oldDistance = selected['distance'][0]
        features = getSensorNames(selected['geometry'], sensorNames)
        oldMSE = getMSE(features, df)

    def calculateRewardsAdding(index, row):
        print('index ', index + 1, 'of', len(unselectedIB))
        extractedRow = gpd.GeoDataFrame([row], geometry='geometry', crs=unselectedIB.crs)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, extractedRow], ignore_index=True), crs=unselectedIB.crs)
        features = getSensorNames(tempSHP['geometry'], sensorNames)
        newMSE = getMSE(features, df)
        newDistance = unselectedIB['distanceIfAdded'][index]
        reward = (oldDistance - newDistance) * (oldMSE - newMSE)
        return index, reward

    indexesRewards = Parallel(n_jobs=-1)(delayed(calculateRewardsAdding)
                                            (index, row) for index, row in unselectedIB.iterrows())

    indexesRewards.sort(key=lambda x: x[0])
    indexes, rewards = zip(*indexesRewards)
    maxReward = max(rewards)
    maxIndex = rewards.index(maxReward)

    distanceOfBest = unselectedIB['distanceIfAdded'][maxIndex]
    rowToMove = unselectedIB.loc[[maxIndex]].copy()
    rowToMove.rename(columns={'distanceIfAdded': 'distance'}, inplace=True)
    selected.loc[:, 'distance'] = distanceOfBest
    rowToMovePoint = rowToMove['geometry'].iloc[0]
    unselected = unselected[unselected['geometry'] != rowToMovePoint]
    selected = gpd.GeoDataFrame(pd.concat([selected, rowToMove], ignore_index=True), crs=unselectedIB.crs)
    return unselected, selected


# Remove the best hoverpoint from selected, best = max((oldDistance - newDistance) * (oldMSE - newMSE))
def remBestHP(unselected, selected):
    print('REMOVE BEST')
    rewards = [float('-inf')] * len(selected)
    if selected.empty:  # never happens in actual greedy alg
        return unselected, selected
    elif len(selected) == 1:
        return remRandomHP(unselected, selected)
    else:
        oldDistance = selected['distance'][0]
        points = selected['geometry'].copy()
        features = getSensorNames(points, sensorNames)
        oldMSE = getMSE(features, df)

    def calculateRewardsRemoving(index, row):
        print('index ', index + 1, 'of', len(selected))
        tempSHP = selected.drop(index=index)
        tempSHP = tempSHP.reset_index(drop=True)
        tempSHP = findMinTravelDistance(tempSHP)
        features = getSensorNames(tempSHP['geometry'], sensorNames)
        newMSE = getMSE(features, df)
        newDistance = tempSHP['distance'][0]
        return index, newMSE, newDistance

    indexMSEDistance = Parallel(n_jobs=-1)(delayed(calculateRewardsRemoving)
                                             (index, row) for index, row in selected.iterrows())
    for value in indexMSEDistance:
        index, newMSE, newDistance = value
        reward = (oldDistance - newDistance) * (oldMSE - newMSE)
        rewards[index] = reward
        if reward == max(rewards):
            distanceOfBest = newDistance

    maxReward = max(rewards)
    maxIndex = rewards.index(maxReward)
    selected.loc[:, 'distance'] = distanceOfBest
    rowToMove = selected.loc[[maxIndex]].copy()
    UHP = gpd.GeoDataFrame(pd.concat([unselected, rowToMove], ignore_index=True), crs=unselected.crs)
    selected = selected.drop(maxIndex).reset_index(drop=True)
    return UHP, selected


arProb = 0  # probability of adding (0) and removing (1)
rbProb = 1  # probability of random (1) and best (0)
L = 60  # number of loops
loopCount = 0
pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf)
minMSE = 1000

# create a new plot of mse over loop iterations
fig2, ax2 = plt.subplots()
x = []
y = []
line, = ax2.plot(x, y)
ax2.set_xlabel('Loop iteration')
ax2.set_ylabel('MSE')

def updateMSEPlot(newX, newY):
    x.append(newX)
    y.append(newY)
    line.set_data(x, y)
    ax2.relim()
    ax2.autoscale_view()
    fig2.canvas.draw()


# Make sure every hoverpoint in hp is accounted for in uhp and shp
if len(UHP_gdf) + len(SHP_gdf) != len(HP_gdf):
    print('ERROR: SELECTED + UNSELECTED != HP')
    exit(1)
while rbProb > 0:
    loopCount += 1
    print("Loop iteration ", loopCount, ' of ', L)
    raProb = rbProb * (1 - arProb)  # random add
    rrProb = rbProb * arProb  # random remove
    baProb = (1 - rbProb) * (1 - arProb)  # best add
    brProb = (1 - rbProb) * arProb  # best remove
    randomNumber = (random.random())
    if randomNumber < raProb:
        UHP_gdf, SHP_gdf = addRandomHP(UHP_gdf, pointsInBudget, SHP_gdf)
    elif randomNumber < raProb + rrProb:
        UHP_gdf, SHP_gdf = remRandomHP(UHP_gdf, SHP_gdf)
    elif randomNumber < raProb + rrProb + baProb:
        UHP_gdf, SHP_gdf = addBestHP(UHP_gdf, pointsInBudget, SHP_gdf)
    else:
        UHP_gdf, SHP_gdf = remBestHP(UHP_gdf, SHP_gdf)

    UHP_gdf = UHP_gdf.reset_index(drop=True)
    SHP_gdf = SHP_gdf.reset_index(drop=True)
    rbProb = rbProb - 1 / L
    features = getSensorNames(SHP_gdf['geometry'], sensorNames)
    mse = getMSE(features, df)
    if mse < minMSE:
        minMSE = mse
        bestSHP = SHP_gdf.copy()
        iterationOfBest = loopCount
    if SHP_gdf.empty:
        totalDistance = 0
    else:
        totalDistance = SHP_gdf['distance'][0]

    print('Total Distance Traveled: ', totalDistance)
    print('Total Number of Hover Points Visited: ', len(SHP_gdf))
    print('mse: ', mse)
    updateMSEPlot(loopCount, mse)
    pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf)

    if len(UHP_gdf) == 0:
        arProb = totalDistance / maxDistance
    else:
        arProb = (totalDistance / maxDistance) * (
                1 - (len(pointsInBudget) / len(UHP_gdf)))  # as we approach budget, more likely to remove
    if len(UHP_gdf) + len(SHP_gdf) != len(HP_gdf):
        print('ERROR: SELECTED + UNSELECTED != HP')
        print(len(SHP_gdf), '+', len(UHP_gdf), ' != ', len(HP_gdf))
        break

bestSHP = findMinTravelDistance(bestSHP)
ax2.scatter(iterationOfBest, minMSE, color='red', label='Lowest MSE')
print(f"lowest mse: {minMSE}")
plotPath(ax, bestSHP)
endTime = time.time()
runTime = endTime - startTime
minutes = int(runTime // 60)
seconds = int(runTime % 60)
print(f"Runtime: {minutes} minutes {seconds} seconds")
plt.show()
