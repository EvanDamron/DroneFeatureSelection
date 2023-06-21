# epsilon squared greedy algorithm to find the optimal set of hover points to predict the other features without
# violating battery-of-drone budget

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
sensorsWithData = set(df.columns)
sensorNames = {point: sensors for point, sensors in sensorNames.items() if any(sensor in sensorsWithData for sensor in sensors)}
pointsToRemove = [point for point in sensorNames.keys() if not sensorNames[point]]
sensorNames = {point: sensorNames[point] for point in sensorNames.keys() if sensorNames[point]}
HP_gdf = HP_gdf.loc[HP_gdf.geometry.isin(sensorNames.keys())]

SHP_gdf = gpd.GeoDataFrame()
SHP_gdf['geometry'] = None
SHP_gdf = SHP_gdf.set_geometry('geometry')
UHP_gdf = HP_gdf.copy()
SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
UHP_gdf.crs = 'EPSG:3857'
outOfBudget = False

# made-up drone specs
energyBudget = 200000  # battery life of drone in joules
joulesPerMeter = 50  # Joules burned per meter traveled
joulesPerSecond = 40  # Joules burned per second of hovering and collecting data
timeOf1 = 30  # amount of seconds it takes to collect data at HP in range of 1 sensor
timeOf2 = 60  # amount of seconds it takes to collect data at HP in range of 2 sensors
timeOf3 = 90  # amount of seconds it takes to collect data at HP in range of 3 sensors

#get the total energy cost to travel to all the selected hover points and collect data from them
def getEnergy(selected):
    energy = 0
    selected, distance = findMinTravelDistance(selected)
    energy += distance * joulesPerMeter
    for hoverPoint in selected['geometry']:
        numCorrespondingSensors = len(sensorNames[hoverPoint])
        if numCorrespondingSensors == 1:
            energy += timeOf1 * joulesPerSecond
        elif numCorrespondingSensors == 2:
            energy += timeOf2 * joulesPerSecond
        elif numCorrespondingSensors == 3:
            energy += timeOf3 * joulesPerSecond
        else:
            print('ERROR: HOVERPOINT CORRESPONDS TO MORE THAN 3 SENSORS')
            exit(1)
    selected['energy'] = energy
    return selected, distance

# determine what points can be visited without violating Energy budget
def getPointsInBudget(unselected, selected):
    inBudgetHP = unselected.copy()
    def calculateEnergyIfAdded(index, row):
        extractedRow = gpd.GeoDataFrame([row], geometry='geometry', crs=unselected.crs)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, extractedRow], ignore_index=True), crs=unselected.crs)
        tempSHP, _ = getEnergy(tempSHP)
        return index, tempSHP['energy'][0]

    indexesEnergy = Parallel(n_jobs=-1)(delayed(calculateEnergyIfAdded)(index, row) for index, row in inBudgetHP.iterrows())
    indexesEnergy.sort(key=lambda x: x[0])
    for value in indexesEnergy:
        index, energy = value
        if energy > energyBudget:
            inBudgetHP = inBudgetHP.drop(index)
        else:
            inBudgetHP.at[index, 'energyIfAdded'] = energy
    if 'energy' in inBudgetHP.columns:
        inBudgetHP.drop('energy', axis=1, inplace=True)
    inBudgetHP = inBudgetHP.reset_index(drop=True)
    return inBudgetHP

# Adds a random hoverpoint that is in budget (IB)
def addRandomHP(unselected, unselectedIB, selected):
    print('ADD RANDOM')
    if unselectedIB.empty:
        print('No more hover points within budget')
        return unselected, selected
    randomRow = unselectedIB.sample(n=1)
    randomRow.rename(columns={'energyIfAdded': 'energy'}, inplace=True)
    randomRow = randomRow.reset_index(drop=True)
    pointAdded = randomRow.iloc[0]['geometry']
    unselected = unselected[unselected['geometry'] != pointAdded].reset_index(drop=True)
    SHP = gpd.GeoDataFrame(pd.concat([selected, randomRow], ignore_index=True), crs=selected.crs)
    SHP['energy'] = randomRow['energy'][0]
    return unselected, SHP


# Remove a random hover point from selected
def remRandomHP(unselected, selected):
    print('REMOVE RANDOM')
    if selected.empty:
        print('selected is empty')
        return selected
    rowToMove = selected.sample(n=1)
    del rowToMove['energy']
    UHP = gpd.GeoDataFrame(pd.concat([unselected, rowToMove], ignore_index=True), crs=unselected.crs)
    # if 'energy' in UHP.columns:
    #     UHP.drop('energy', axis=1, inplace=True)
    selected = selected.drop(index=rowToMove.index)
    return UHP, selected


# Add the best in-budget-hoverpoint to selected, best = max((oldEnergy - newEnergy) * (oldMSE - newMSE))

def addBestHP(unselected, unselectedIB, selected):
    print("ADD BEST")
    if unselectedIB.empty:
        print('No more hover points within budget')
        return unselected, selected
    if selected.empty:  # never happens in actual greedy alg
        oldEnergy = 0
        oldMSE = 999999999
    else:
        oldEnergy = selected['energy'][0]
        features = getSensorNames(selected['geometry'], sensorNames)
        oldMSE = getMSE(features, df)

    def calculateRewardsAdding(index, row):
        print('index ', index + 1, 'of', len(unselectedIB))
        extractedRow = gpd.GeoDataFrame([row], geometry='geometry', crs=unselectedIB.crs)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, extractedRow], ignore_index=True), crs=unselectedIB.crs)
        features = getSensorNames(tempSHP['geometry'], sensorNames)
        newMSE = getMSE(features, df)
        newEnergy = unselectedIB['energyIfAdded'][index]
        reward = (oldEnergy - newEnergy) * (oldMSE - newMSE)
        return index, reward

    indexesRewards = Parallel(n_jobs=-1)(delayed(calculateRewardsAdding)
                                            (index, row) for index, row in unselectedIB.iterrows())

    indexesRewards.sort(key=lambda x: x[0])
    indexes, rewards = zip(*indexesRewards)
    maxReward = max(rewards)
    maxIndex = rewards.index(maxReward)

    rowToMove = unselectedIB.loc[[maxIndex]].copy()
    rowToMove.rename(columns={'energyIfAdded': 'energy'}, inplace=True)
    rowToMove = rowToMove.reset_index(drop=True)
    selected.loc[:, 'energy'] = rowToMove['energy'][0]
    rowToMovePoint = rowToMove['geometry'].iloc[0]
    unselected = unselected[unselected['geometry'] != rowToMovePoint]
    selected = gpd.GeoDataFrame(pd.concat([selected, rowToMove], ignore_index=True), crs=unselectedIB.crs)
    return unselected, selected


# Remove the best hoverpoint from selected, best = max((oldDistance - newDistance) * (oldMSE - newMSE))
def remBestHP(unselected, selected):
    print('REMOVE BEST')
    rewards = [float('-inf')] * len(selected)
    if selected.empty:  # shouldn't happen in actual greedy alg, used for testing
        return unselected, selected
    elif len(selected) == 1:
        return remRandomHP(unselected, selected)
    else:
        oldEnergy = selected['energy'][0]
        features = getSensorNames(selected['geometry'], sensorNames)
        oldMSE = getMSE(features, df)

    def calculateRewardsRemoving(index):
        print('index ', index + 1, 'of', len(selected))
        tempSHP = selected.drop(index=index).reset_index(drop=True)
        tempSHP, _ = getEnergy(tempSHP)
        features = getSensorNames(tempSHP['geometry'], sensorNames)
        newMSE = getMSE(features, df)
        newEnergy = tempSHP['energy'][0]
        return index, newMSE, newEnergy

    indexMSEEnergy = Parallel(n_jobs=-1)(delayed(calculateRewardsRemoving)
                                             (index) for index, _ in selected.iterrows())
    for value in indexMSEEnergy:
        index, newMSE, newEnergy = value
        reward = (oldEnergy - newEnergy) * (oldMSE - newMSE)
        rewards[index] = reward
        if reward == max(rewards):
            energyOfBest = newEnergy

    maxReward = max(rewards)
    maxIndex = rewards.index(maxReward)
    selected.loc[:, 'energy'] = energyOfBest
    rowToMove = selected.loc[[maxIndex]].copy()
    UHP = gpd.GeoDataFrame(pd.concat([unselected, rowToMove], ignore_index=True), crs=unselected.crs)
    selected = selected.drop(maxIndex).reset_index(drop=True)
    return UHP, selected


arProb = 0  # probability of adding (0) and removing (1)
rbProb = 1  # probability of random (1) and best (0)
L = 50 # number of loops
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
while loopCount < L:
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
        totalEnergy = 0
    else:
        totalEnergy = SHP_gdf['energy'][0]

    print(f"This set of hoverpoints requires {totalEnergy} Joules out of the"
          f" {energyBudget} Joules in the drone's battery")
    print('Total Number of Hover Points Visited: ', len(SHP_gdf))
    print(f"current mse: {mse}, lowest mse yet: {minMSE}")
    updateMSEPlot(loopCount, mse)
    pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf)

    if len(UHP_gdf) == 0:
        arProb = totalEnergy / energyBudget
    else:
        arProb = (energyBudget / energyBudget) * (
                1 - (len(pointsInBudget) / len(UHP_gdf)))  # as we approach budget, more likely to remove
    if len(UHP_gdf) + len(SHP_gdf) != len(HP_gdf):
        print('ERROR: SELECTED + UNSELECTED != HP')
        print(len(SHP_gdf), '+', len(UHP_gdf), ' != ', len(HP_gdf))
        break

bestSHP, distanceOfBest = getEnergy(bestSHP)
ax2.scatter(iterationOfBest, minMSE, color='red', label='Lowest MSE')
print(f"The best path's mse was {minMSE}, and it was found on the {iterationOfBest}th iteration")
print(f"It consumed {bestSHP['energy'][0]} joules and traveled {distanceOfBest} meters")
print(f"sensors used: {getSensorNames(bestSHP['geometry'], sensorNames)}")
plotPath(ax, bestSHP)
endTime = time.time()
runTime = endTime - startTime
minutes = int(runTime // 60)
seconds = int(runTime % 60)
print(f"Runtime: {minutes} minutes {seconds} seconds")
plt.show()
