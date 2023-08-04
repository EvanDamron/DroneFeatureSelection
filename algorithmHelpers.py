from mapping import processSHP, findMinTravelDistance, generateSensorsUniformRandom, getSensorNames, minSetCover
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from ML import getMSE, processData
import random
from joblib import Parallel, delayed
import time
import numpy as np





def processFiles(height, communicationRadius, minSet, generate):
    positionFilePath = 'CAF_Sensor_Dataset_2/CAF_sensors.shp'
    dataFolderPath = 'CAF_Sensor_Dataset_2/caf_sensors/Hourly'
    df = processData(dataFolderPath)
    if generate:
        fig, ax, hoverPoints, sensorNames = generateSensorsUniformRandom(height, communicationRadius, df)
    else:
        fig, ax, hoverPoints, sensorNames = processSHP(positionFilePath, height, communicationRadius)
    if minSet:
        sensorNames, hoverPoints = minSetCover(sensorNames, hoverPoints)
    hoverPoints.plot(ax=ax, color='yellow', markersize=10, alpha=1)
    # hoverPoints = hoverPoints.sample(n=20)  # Use this to test, reduces runtime
    sensorsWithData = set(df.columns)
    filteredDictionary = {}
    for point, sensors in sensorNames.items():
        filteredSensors = [sensor for sensor in sensors if sensor in sensorsWithData]
        if filteredSensors:
            filteredDictionary[point] = filteredSensors
    sensorNames = filteredDictionary
    hoverPoints = hoverPoints.loc[hoverPoints.geometry.isin(sensorNames.keys())]
    if len(hoverPoints) != len(sensorNames.values()):
        print(f"points in hpgdf = {len(hoverPoints.columns)}, sensorNames values = {len(sensorNames.values())}")
        print('ERROR')
    selected = gpd.GeoDataFrame()
    selected['geometry'] = None
    selected = selected.set_geometry('geometry')
    unselected = hoverPoints.copy()
    selected.crs = 'EPSG:3857'  # pseudo-mercator
    unselected.crs = 'EPSG:3857'
    return fig, ax, hoverPoints, unselected, selected, sensorNames, df


# get the total energy cost to travel to all the selected hover points and collect data from them
def getEnergy(selected, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate):
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
def getPointsInBudget(unselected, selected, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond, dataSize, transferRate):
    inBudgetHP = unselected.copy()

    def calculateEnergyIfAdded(index, row):
        extractedRow = gpd.GeoDataFrame([row], geometry='geometry', crs=unselected.crs)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, extractedRow], ignore_index=True), crs=unselected.crs)
        tempSHP, _ = getEnergy(tempSHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
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
        maxReward = max(rewards)
        if maxReward == 0:
            nonZeroRewards = [reward for reward in rewards if reward != 0]
            maxReward = max(nonZeroRewards)
        indexOfBest = rewards.index(maxReward)

    rowToMove = unselectedIB.loc[[indexOfBest]].copy()
    rowToMove.rename(columns={'energyIfAdded': 'energy'}, inplace=True)
    rowToMove = rowToMove.reset_index(drop=True)
    selected['energy'] = rowToMove['energy'][0]
    rowToMovePoint = rowToMove['geometry'].iloc[0]
    unselected = unselected[unselected['geometry'] != rowToMovePoint]
    selected = gpd.GeoDataFrame(pd.concat([selected, rowToMove], ignore_index=True), crs=unselectedIB.crs)
    return unselected, selected


# Remove the best hover-point from selected
def remBestHP(unselected, selected, rewardMode, thresholdFraction, sensorNames, df, energyWeight, joulesPerMeter, joulesPerSecond, dataSize, transferRate):
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
        tempSHP, _ = getEnergy(tempSHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
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

def writeResults(finalSHP, finalIteration, finalDistance, finalMSE, sensorNames, outputTextName):
    with open(outputTextName, mode='w', encoding='utf-8') as file:
        file.write(f"final energy of best = {finalSHP['energy'][0]} joules\n")
        file.write(f"The best path's mse was {finalMSE}, and it was found on the {finalIteration}th iteration\n")
        file.write(f"It consumed {finalSHP['energy'][0]} joules and traveled {finalDistance} meters\n")
        sensors = getSensorNames(finalSHP['geometry'], sensorNames)
        file.write(f"sensors used: {sensors} ({len(sensors)})\n")
        file.write(f"best SHP:\n{finalSHP}")


def printTime(startTime):
    endTime = time.time()
    runTime = endTime - startTime
    minutes = int(runTime // 60)
    seconds = int(runTime % 60)
    print(f"Runtime: {minutes} minutes {seconds} seconds")