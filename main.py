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
import numpy as np


np.random.seed(42)
random.seed(42)
# made-up drone specs
energyBudget = 60000  # battery life of drone in joules
joulesPerMeter = 10  # Joules burned per meter traveled
joulesPerSecond = 40  # Joules burned per second of hovering and collecting data
dataSize = 100   # In Mb
transferRate = 9  # In Mb/s, 9 is Wifi 4 standard

def processFiles(communicationRadius):
    positionFilePath = 'CAF_Sensor_Dataset_2/CAF_sensors.shp'
    dataFolderPath = 'CAF_Sensor_Dataset_2/caf_sensors/Hourly'

    # HP: Hover Points S:selected U:unselected
    ax, HP_gdf, sensorNames = processSHP(positionFilePath, communicationRadius)
    # HP_gdf = HP_gdf.sample(n=20)  # Use this to test, reduces runtime
    df = processData(dataFolderPath)
    sensorsWithData = set(df.columns)
    filteredDictionary = {}
    for point, sensors in sensorNames.items():
        filteredSensors = [sensor for sensor in sensors if sensor in sensorsWithData]
        if filteredSensors:
            filteredDictionary[point] = filteredSensors
    sensorNames = filteredDictionary
    HP_gdf = HP_gdf.loc[HP_gdf.geometry.isin(sensorNames.keys())]
    if len(HP_gdf) != len(sensorNames.values()):
        print(f"points in hpgdf = {len(HP_gdf.columns)}, sensorNames values = {len(sensorNames.values())}")
        print('ERROR')
    SHP_gdf = gpd.GeoDataFrame()
    SHP_gdf['geometry'] = None
    SHP_gdf = SHP_gdf.set_geometry('geometry')
    UHP_gdf = HP_gdf.copy()
    SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
    UHP_gdf.crs = 'EPSG:3857'
    return ax, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df


# get the total energy cost to travel to all the selected hover points and collect data from them
def getEnergy(selected, sensorNames):
    energy = 0
    selected, distance = findMinTravelDistance(selected)
    energy += distance * joulesPerMeter
    for hoverPoint in selected['geometry']:
        numCorrespondingSensors = len(sensorNames[hoverPoint])
        timeToTransfer = numCorrespondingSensors * dataSize / transferRate
        energy += timeToTransfer * joulesPerSecond
    selected['energy'] = energy
    return selected, distance


# determine what points can be visited without violating Energy budget
def getPointsInBudget(unselected, selected, sensorNames):
    inBudgetHP = unselected.copy()

    def calculateEnergyIfAdded(index, row):
        extractedRow = gpd.GeoDataFrame([row], geometry='geometry', crs=unselected.crs)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, extractedRow], ignore_index=True), crs=unselected.crs)
        tempSHP, _ = getEnergy(tempSHP, sensorNames)
        return index, tempSHP['energy'][0]

    indexesEnergy = Parallel(n_jobs=-1)(
        delayed(calculateEnergyIfAdded)(index, row) for index, row in inBudgetHP.iterrows())
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

def addBestHP(unselected, unselectedIB, selected, rewardMode, thresholdFraction, sensorNames, df, energyWeight):
    print("ADD BEST")
    rewards = []
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
        # print('index ', index + 1, 'of', len(unselectedIB))
        extractedRow = gpd.GeoDataFrame([row], geometry='geometry', crs=unselectedIB.crs)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, extractedRow], ignore_index=True), crs=unselectedIB.crs)
        features = getSensorNames(tempSHP['geometry'], sensorNames)
        newMSE = getMSE(features, df)
        newEnergy = unselectedIB['energyIfAdded'][index]
        return index, newMSE, newEnergy

    indexMSEEnergy = Parallel(n_jobs=-1)(delayed(calculateRewardsAdding)
                                         (index, row) for index, row in unselectedIB.iterrows())
    indexMSEEnergy = sorted(indexMSEEnergy, key=lambda x: x[0])

    if rewardMode == "MSE":
        for value in indexMSEEnergy:
            index, newMSE, newEnergy = value
            reward = (oldMSE - newMSE)
            rewards.append(reward)
    elif rewardMode == "ENERGY":
        for value in indexMSEEnergy:
            index, newMSE, newEnergy = value
            reward = (oldEnergy - newEnergy)
            rewards.append(reward)
    elif rewardMode == "BOTH":
        for value in indexMSEEnergy:
            index, newMSE, newEnergy = value
            reward = (oldMSE - newMSE) * (oldEnergy - newEnergy)
            rewards.append(reward)
    elif rewardMode == "NORMALIZED":
        minMSE = min(value[1] for value in indexMSEEnergy)
        maxMSE = max(value[1] for value in indexMSEEnergy)
        minEnergy = min(value[2] for value in indexMSEEnergy)
        maxEnergy = max(value[2] for value in indexMSEEnergy)
        for index, newMSE, newEnergy in indexMSEEnergy:
            normalizedNewMSE = (newMSE - minMSE) / (maxMSE - minMSE)
            normalizedNewEnergy = (newEnergy - minEnergy) / (maxEnergy - minEnergy)
            reward = 1 - (normalizedNewEnergy * energyWeight + normalizedNewMSE) / 2
            rewards.append(reward)
    elif rewardMode == "THRESHOLD":
        indexOfBest = -1
        minEnergy = 999999999
        for value in indexMSEEnergy:
            index, newMSE, newEnergy = value
            if (newMSE * thresholdFraction) < oldMSE:
                if newEnergy < minEnergy:
                    indexOfBest = index
                    minEnergy = newEnergy
        if indexOfBest < 0:
            print("NO UNSELECTED FEATURES IN BUDGET AND ABOVE ACCURACY THRESHOLD")
            return unselected, selected
    if rewards:
        indexOfBest = rewards.index(max(rewards))

    rowToMove = unselectedIB.loc[[indexOfBest]].copy()
    rowToMove.rename(columns={'energyIfAdded': 'energy'}, inplace=True)
    rowToMove = rowToMove.reset_index(drop=True)
    selected['energy'] = rowToMove['energy'][0]
    rowToMovePoint = rowToMove['geometry'].iloc[0]
    unselected = unselected[unselected['geometry'] != rowToMovePoint]
    selected = gpd.GeoDataFrame(pd.concat([selected, rowToMove], ignore_index=True), crs=unselectedIB.crs)
    return unselected, selected


# Remove the best hover-point from selected
def remBestHP(unselected, selected, rewardMode, thresholdFraction, sensorNames, df, energyWeight):
    print('REMOVE BEST')
    rewards = []
    if selected.empty:  # shouldn't happen in actual greedy alg, used for testing
        return unselected, selected
    elif len(selected) == 1:
        return remRandomHP(unselected, selected)
    else:
        oldEnergy = selected['energy'][0]
        features = getSensorNames(selected['geometry'], sensorNames)
        oldMSE = getMSE(features, df)

    def calculateRewardsRemoving(index):
        # print('index ', index + 1, 'of', len(selected))
        tempSHP = selected.drop(index=index).reset_index(drop=True)
        tempSHP, _ = getEnergy(tempSHP, sensorNames)
        features = getSensorNames(tempSHP['geometry'], sensorNames)
        newMSE = getMSE(features, df)
        newEnergy = tempSHP['energy'][0]
        return index, newMSE, newEnergy

    indexMSEEnergy = Parallel(n_jobs=-1)(delayed(calculateRewardsRemoving)
                                         (index) for index, _ in selected.iterrows())
    indexMSEEnergy = sorted(indexMSEEnergy, key=lambda x: x[0])

    if rewardMode == "MSE":
        for value in indexMSEEnergy:
            index, newMSE, newEnergy = value
            reward = (oldMSE - newMSE)
            rewards.append(reward)
    elif rewardMode == "ENERGY":
        for value in indexMSEEnergy:
            index, newMSE, newEnergy = value
            reward = (oldEnergy - newEnergy)
            rewards.append(reward)
    elif rewardMode == "BOTH":
        for value in indexMSEEnergy:
            index, newMSE, newEnergy = value
            reward = (oldMSE - newMSE) * (oldEnergy - newEnergy)
            rewards.append(reward)
    elif rewardMode == "NORMALIZED":
        minMSE = min(value[1] for value in indexMSEEnergy)
        maxMSE = max(value[1] for value in indexMSEEnergy)
        minEnergy = min(value[2] for value in indexMSEEnergy)
        maxEnergy = max(value[2] for value in indexMSEEnergy)
        for index, newMSE, newEnergy in indexMSEEnergy:
            normalizedNewMSE = (newMSE - minMSE) / (maxMSE - minMSE)
            normalizedNewEnergy = (newEnergy - minEnergy) / (maxEnergy - minEnergy)
            reward = 1 - (normalizedNewEnergy * energyWeight + normalizedNewMSE) / 2
            rewards.append(reward)
    elif rewardMode == "THRESHOLD":
        indexOfBest = -1
        energyOfBest = 999999999
        for value in indexMSEEnergy:
            index, newMSE, newEnergy = value
            if (newMSE * thresholdFraction) < oldMSE:
                if newEnergy < energyOfBest:
                    indexOfBest = index
                    energyOfBest = newEnergy
        if indexOfBest < 0:
            print("NO UNSELECTED FEATURES IN BUDGET AND ABOVE ACCURACY THRESHOLD")
            return unselected, selected
    if rewards:
        indexOfBest = rewards.index(max(rewards))
        for index, newMSE, newEnergy in indexMSEEnergy:
            if index == indexOfBest:
                energyOfBest = newEnergy

    selected['energy'] = energyOfBest
    rowToMove = selected.loc[[indexOfBest]].copy()
    UHP = gpd.GeoDataFrame(pd.concat([unselected, rowToMove], ignore_index=True), crs=unselected.crs)
    selected = selected.drop(indexOfBest).reset_index(drop=True)
    return UHP, selected


def createMSEPlot():
    fig2, ax2 = plt.subplots()
    x = []
    y = []
    line, = ax2.plot(x, y)
    ax2.set_xlabel('Loop iteration')
    ax2.set_ylabel('MSE')
    return x, y, line, ax2, fig2


def updateMSEPlot(newX, newY, ax, fig, x, y, line):
    x.append(newX)
    y.append(newY)
    line.set_data(x, y)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()


def printResults(finalSHP, finalIteration, finalDistance, finalMSE, sensorNames):
    print(f"final energy of best = {finalSHP['energy'][0]} joules")
    print(f"The best path's mse was {finalMSE}, and it was found on the {finalIteration}th iteration")
    print(f"It consumed {finalSHP['energy'][0]} joules and traveled {finalDistance} meters")
    sensors = getSensorNames(finalSHP['geometry'], sensorNames)
    print(f"sensors used: {sensors} ({len(sensors)})")
    print(f"best SHP:\n{finalSHP}")


def printTime(startTime):
    endTime = time.time()
    runTime = endTime - startTime
    minutes = int(runTime // 60)
    seconds = int(runTime % 60)
    print(f"Runtime: {minutes} minutes {seconds} seconds")


def epsilonGreedy(numLoops, startTime, addRewardMode="MSE", remRewardMode="MSE", thresholdFraction=0.95, energyWeight=1,
                  communicationRadius=70):
    ax, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df = processFiles(communicationRadius)
    print(f"Total number of Hoverpoints: {len(HP_gdf)}")
    x, y, line, ax2, fig2 = createMSEPlot()
    print(
        f"EPSILON-GREEDY ALGORITHM WITH REWARD MODE {addRewardMode} (ADD) {remRewardMode} (REMOVE), AND ENERGY WEIGHT {energyWeight}")
    arProb = 0  # probability of adding (0) and removing (1)
    rbProb = 1  # probability of random (1) and best (0)
    loopCount = 0
    pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf, sensorNames)
    print(pointsInBudget)
    minMSE = 1000

    # Make sure every hover-point in hp is accounted for in uhp and shp
    if len(UHP_gdf) + len(SHP_gdf) != len(HP_gdf):
        print('ERROR: SELECTED + UNSELECTED != HP')
        exit(1)
    while loopCount < numLoops:
        loopCount += 1
        # energyWeight = 1 - (loopCount / numLoops)
        print("Loop iteration ", loopCount, ' of ', numLoops)
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
            UHP_gdf, SHP_gdf = addBestHP(unselected=UHP_gdf, unselectedIB=pointsInBudget, selected=SHP_gdf,
                                         rewardMode=addRewardMode, thresholdFraction=thresholdFraction,
                                         sensorNames=sensorNames, df=df, energyWeight=energyWeight)
        else:
            UHP_gdf, SHP_gdf = remBestHP(UHP_gdf, SHP_gdf, rewardMode=remRewardMode, sensorNames=sensorNames,
                                         thresholdFraction=thresholdFraction, df=df, energyWeight=energyWeight)

        UHP_gdf = UHP_gdf.reset_index(drop=True)
        SHP_gdf = SHP_gdf.reset_index(drop=True)
        rbProb = rbProb - 1 / numLoops
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
        updateMSEPlot(newX=loopCount, newY=mse, ax=ax2, fig=fig2, x=x, y=y, line=line)
        pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf, sensorNames=sensorNames)

        if len(UHP_gdf) == 0:
            arProb = totalEnergy / energyBudget
        else:
            arProb = (energyBudget / energyBudget) * (
                    1 - (len(pointsInBudget) / len(UHP_gdf)))  # as we approach budget, more likely to remove
        if len(UHP_gdf) + len(SHP_gdf) != len(HP_gdf):
            print('ERROR: SELECTED + UNSELECTED != HP')
            print(len(SHP_gdf), '+', len(UHP_gdf), ' != ', len(HP_gdf))
            break
    # get the cheapest path of best shp, because TSP is heuristic and gives different paths, some better than others
    minEnergy = 999999999999
    for i in range(10):
        tempBestSHP = bestSHP.copy()
        tempBestSHP, distance = getEnergy(tempBestSHP, sensorNames)
        print(f"energy of best {i}: {tempBestSHP['energy'][0]}")
        if tempBestSHP['energy'][0] < minEnergy:
            minEnergy = tempBestSHP['energy'][0]
            bestSHP = tempBestSHP.copy()
            distanceOfBest = distance
    print(f"EPSILON-GREEDY ALGORITHM WITH REWARD MODE {addRewardMode} (ADD) {remRewardMode} (REMOVE), AND ENERGY WEIGHT {energyWeight}")
    printResults(bestSHP, iterationOfBest, distanceOfBest, minMSE, sensorNames)
    ax2.scatter(iterationOfBest, minMSE, color='red', zorder=10)
    plotPath(ax, bestSHP)
    bestSHP.plot(ax=ax, color='red', markersize=10, alpha=1)
    printTime(startTime)
    plt.show()


def SFS(rewardMode="MSE", thresholdFraction=0.95, energyWeight=1, communicationRadius=70):
    print('SEQUENTIAL FORWARD SELECTION')
    ax, HP_gdf, UHP, SHP, sensorNames, df = processFiles(communicationRadius)
    x, y, line, ax2, fig2 = createMSEPlot()
    minMSE = 10
    loopCount = 0
    while True:
        # energyWeight = 1 - (loopCount / 10)
        loopCount += 1
        pointsInBudget = getPointsInBudget(UHP, SHP, sensorNames)
        if pointsInBudget.empty:
            bestSHP, distanceOfBest = getEnergy(bestSHP, sensorNames)
            while bestSHP['energy'][0] > energyBudget:
                bestSHP, distanceOfBest = getEnergy(bestSHP, sensorNames)
            bestSHP = bestSHP.reset_index(drop=True)
            printResults(finalSHP=bestSHP, finalMSE=minMSE, finalIteration=iterationOfBest,
                         finalDistance=distanceOfBest,
                         sensorNames=sensorNames)
            break
        UHP, SHP = addBestHP(unselected=UHP, unselectedIB=pointsInBudget, selected=SHP, rewardMode=rewardMode,
                             thresholdFraction=thresholdFraction, df=df, sensorNames=sensorNames,
                             energyWeight=energyWeight)
        SHP = SHP.reset_index(drop=True)
        UHP = UHP.reset_index(drop=True)
        features = getSensorNames(SHP['geometry'], sensorNames)
        mse = getMSE(features, df)
        updateMSEPlot(newX=loopCount, newY=mse, ax=ax2, fig=fig2, x=x, y=y, line=line)
        SHP, distance = getEnergy(SHP, sensorNames)
        energy = SHP['energy'][0]
        if mse < minMSE:
            minMSE = mse
            bestSHP = SHP
            iterationOfBest = loopCount
            print(f"MSE = {minMSE}, energy = {energy} / {energyBudget} joules")
        else:
            print(f"MSE = {mse}, energy = {energy} / {energyBudget} joules")

    ax2.scatter(iterationOfBest, minMSE, color='red', zorder=10)
    plotPath(ax, bestSHP)
    bestSHP.plot(ax=ax, color='red', markersize=10, alpha=1)
    # printTime(startTime=startTime)
    plt.show()


def SBS(rewardMode="MSE", thresholdFraction=0.95, energyWeight=1):
    print('SEQUENTIAL BACKWARD SELECTION')
    ax, HP_gdf, UHP, SHP, sensorNames, df = processFiles()
    tempSHP = SHP
    SHP = UHP
    UHP = tempSHP
    x, y, line, ax2, fig2 = createMSEPlot()
    minMSE = 10
    loopCount = 0
    SHP, _ = getEnergy(SHP, sensorNames=sensorNames)
    while True:
        # energyWeight = 1 - (loopCount / 10)
        loopCount += 1

        UHP, SHP = remBestHP(unselected=UHP, selected=SHP, rewardMode=rewardMode,
                             thresholdFraction=thresholdFraction, df=df, sensorNames=sensorNames,
                             energyWeight=energyWeight)
        SHP = SHP.reset_index(drop=True)
        UHP = UHP.reset_index(drop=True)
        features = getSensorNames(SHP['geometry'], sensorNames)
        mse = getMSE(features, df)
        updateMSEPlot(newX=loopCount, newY=mse, ax=ax2, fig=fig2, x=x, y=y, line=line)
        SHP, distance = getEnergy(SHP, sensorNames)
        energy = SHP['energy'][0]

        print(f"MSE = {mse}, energy = {energy} / {energyBudget} joules")
        if energy < energyBudget:
            minMSE = mse
            bestSHP = SHP
            iterationOfBest = loopCount
            bestSHP, distanceOfBest = getEnergy(bestSHP, sensorNames)
            bestSHP = bestSHP.reset_index(drop=True)
            printResults(finalSHP=bestSHP, finalMSE=minMSE, finalIteration=iterationOfBest,
                         finalDistance=distanceOfBest,
                         sensorNames=sensorNames)
            break

    ax2.scatter(iterationOfBest, minMSE, color='red', zorder=10)
    plotPath(ax, bestSHP)
    bestSHP.plot(ax=ax, color='red', markersize=10, alpha=1)
    # printTime(startTime=startTime)
    plt.show()



startTime = time.time()
epsilonGreedy(numLoops=40, startTime=startTime, addRewardMode="NORMALIZED", remRewardMode="NORMALIZED", energyWeight=0)


seconds = int(runTime % 60)
print(f"Runtime: {minutes} minutes {seconds} seconds")
plt.show()
