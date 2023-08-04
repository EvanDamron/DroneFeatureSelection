# epsilon squared greedy algorithm to find the optimal set of hover points to predict the other features without
# violating battery-of-drone budget
# Code Written by Evan Damron, University of Kentucky Computer Science

from mapping import processSHP, findMinTravelDistance, plotPath, getSensorNames, minSetCover
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from ML import getMSE, processData
import random
from joblib import Parallel, delayed
import time
import numpy as np
from algorithmHelpers import processFiles, getEnergy, getPointsInBudget, addRandomHP, remRandomHP, addBestHP, remBestHP, \
    createMSEPlot, updateMSEPlot, printResults, writeResults, printTime

np.random.seed(42)
random.seed(42)


def epsilonGreedy(numLoops, startTime, savePlots=False, pathPlotName="", msePlotName="", outputTextName="", droneHeight=15,
                  addRewardMode="MSE", remRewardMode="MSE", thresholdFraction=0.95, energyWeight=1,
                  communicationRadius=70, energyBudget=60000, joulesPerMeter=10, joulesPerSecond=35, dataSize=250,
                  transferRate=9, minSet=False, generate=False):
    fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df = processFiles(droneHeight, communicationRadius, minSet, generate)
    print(f"Total number of Hoverpoints: {len(HP_gdf)}")
    x, y, line, ax2, fig2 = createMSEPlot()
    print(
        f"EPSILON-GREEDY ALGORITHM WITH REWARD MODE {addRewardMode} (ADD) {remRewardMode} (REMOVE), AND ENERGY WEIGHT {energyWeight}")
    arProb = 0  # probability of adding (0) and removing (1)
    rbProb = 1  # probability of random (1) and best (0)
    loopCount = 0
    pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond,
                                       dataSize, transferRate)
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
                                         thresholdFraction=thresholdFraction, df=df, energyWeight=energyWeight, joulesPerMeter=joulesPerMeter, joulesPerSecond=joulesPerSecond, dataSize=dataSize, transferRate=transferRate)

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
        pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond,
                                           dataSize, transferRate)

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
        tempBestSHP, _ = findMinTravelDistance(tempBestSHP, scramble=True)
        tempBestSHP, distance = getEnergy(tempBestSHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize,
                                          transferRate)
        print(f"energy of best {i}: {tempBestSHP['energy'][0]}")
        if tempBestSHP['energy'][0] < minEnergy:
            minEnergy = tempBestSHP['energy'][0]
            bestSHP = tempBestSHP.copy()
            distanceOfBest = distance
    print(
        f"EPSILON-GREEDY ALGORITHM WITH REWARD MODE {addRewardMode} (ADD) {remRewardMode} (REMOVE), AND ENERGY WEIGHT {energyWeight}")
    printResults(bestSHP, iterationOfBest, distanceOfBest, minMSE, sensorNames)
    ax2.scatter(iterationOfBest, minMSE, color='red', zorder=10)
    plotPath(ax1, bestSHP)
    bestSHP.plot(ax=ax1, color='red', markersize=10, alpha=1)
    printTime(startTime)
    if savePlots:
        writeResults(bestSHP, iterationOfBest, distanceOfBest, minMSE, sensorNames, outputTextName)
        fig1.savefig(pathPlotName, bbox_inches='tight')
        fig2.savefig(msePlotName, bbox_inches='tight')
    else:
        plt.show()


# Max-Reward-Energy Algorithm, instead of reward we use MSE
def MRE(rewardMode="NORMALIZED", thresholdFraction=0.95, energyWeight=1, droneHeight=15, energyBudget=60000,
        joulesPerMeter=10, joulesPerSecond=35, dataSize=250, transferRate=9):
    print('MRE ALGORITHM')
    communicationRadius = 40
    fig1, ax1, HP_gdf, UHP, SHP, sensorNames, df = processFiles(droneHeight, communicationRadius)
    x, y, line, ax2, fig2 = createMSEPlot()
    minMSE = 10
    loopCount = 0
    while True:
        # energyWeight = 1 - (loopCount / 10)
        loopCount += 1
        pointsInBudget = getPointsInBudget(UHP, SHP, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond,
                                           dataSize, transferRate)
        if pointsInBudget.empty:
            bestSHP, distanceOfBest = getEnergy(bestSHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize,
                                                transferRate)
            while bestSHP['energy'][0] > energyBudget:
                bestSHP, _ = findMinTravelDistance(bestSHP, scramble=True)
                bestSHP, distanceOfBest = getEnergy(bestSHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize,
                                                    transferRate)
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
        SHP, distance = getEnergy(SHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
        energy = SHP['energy'][0]
        if mse < minMSE:
            minMSE = mse
            bestSHP = SHP
            iterationOfBest = loopCount
            print(f"MSE = {minMSE}, energy = {energy} / {energyBudget} joules")
        else:
            print(f"MSE = {mse}, energy = {energy} / {energyBudget} joules")
    print('LAST SHP:\n')
    printResults(finalSHP=SHP, finalIteration=loopCount, finalDistance=distance, finalMSE=mse, sensorNames=sensorNames)
    ax2.scatter(iterationOfBest, minMSE, color='red', zorder=10)
    plotPath(ax1, bestSHP)
    bestSHP.plot(ax=ax1, color='red', markersize=10, alpha=1)
    print('BEST SHP:\n')
    printResults(finalSHP=bestSHP, finalIteration=iterationOfBest, finalDistance=distanceOfBest, finalMSE=minMSE,
                 sensorNames=sensorNames)
    # printTime(startTime=startTime)
    plt.show()


def SBS(rewardMode="MSE", thresholdFraction=0.95, energyWeight=1, energyBudget=60000, joulesPerMeter=10,
        joulesPerSecond=35, dataSize=250, transferRate=9):
    print('SEQUENTIAL BACKWARD SELECTION')
    ax, HP_gdf, UHP, SHP, sensorNames, df = processFiles()
    tempSHP = SHP
    SHP = UHP
    UHP = tempSHP
    x, y, line, ax2, fig2 = createMSEPlot()
    minMSE = 10
    loopCount = 0
    SHP, _ = getEnergy(SHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
    while True:
        # energyWeight = 1 - (loopCount / 10)
        loopCount += 1

        UHP, SHP = remBestHP(unselected=UHP, selected=SHP, rewardMode=rewardMode,
                             thresholdFraction=thresholdFraction, df=df, sensorNames=sensorNames,
                             energyWeight=energyWeight, joulesPerMeter=joulesPerMeter, joulesPerSecond=joulesPerSecond, dataSize=dataSize, transferRate=transferRate)
        SHP = SHP.reset_index(drop=True)
        UHP = UHP.reset_index(drop=True)
        features = getSensorNames(SHP['geometry'], sensorNames)
        mse = getMSE(features, df)
        updateMSEPlot(newX=loopCount, newY=mse, ax=ax2, fig=fig2, x=x, y=y, line=line)
        SHP, distance = getEnergy(SHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
        energy = SHP['energy'][0]

        print(f"MSE = {mse}, energy = {energy} / {energyBudget} joules")
        if energy < energyBudget:
            minMSE = mse
            bestSHP = SHP
            iterationOfBest = loopCount
            bestSHP, distanceOfBest = getEnergy(bestSHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize,
                                                transferRate)
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
epsilonGreedy(numLoops=35, savePlots=False, generate=True,
                  startTime=startTime, addRewardMode="NORMALIZED", remRewardMode="NORMALIZED", energyWeight=0)
