# epsilon squared greedy algorithm to find the optimal set of hover points to predict the other features without
# violating battery-of-drone budget
# Code Written by Evan Damron, University of Kentucky Computer Science

from mapping import findMinTravelDistance, plotPath, getSensorNames, addSensorsUniformRandom, getHoverPoints
import pandas as pd
import matplotlib.pyplot as plt
from ML import getMSE, calculateEntropy, discretizeData, getConditionalEntropy, getInformationGain, processData,normalizeData
import random
import time
import numpy as np
from algorithmHelpers import processFiles, getEnergy, getPointsInBudget, addRandomHP, remRandomHP, addBestHP, remBestHP, \
    createMSEPlot, updateMSEPlot, printResults, writeResults, printTime, createIGMSEPlot, updateIGMSEPlot, \
    addBestHybrid, remBestHybrid
import signal
import sys
import geopandas as gpd


def signal_handler(sig, frame):
    print('Ctrl+C pressed! Showing plot...')
    plt.show()  # display the plot
    sys.exit(0)  # exit the program

# fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
def epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
                  numLoops, startTime, savePlots=False, pathPlotName="", msePlotName="", outputTextName="",
                  droneHeight=15, energyWeight=0, communicationRadius=70, energyBudget=40000, joulesPerMeter=10,
                  joulesPerSecond=35, dataSize=100, transferRate=9, minSet=False, generate=False, numSensors=37,
                  addToOriginal=True, exhaustive=True):
    # fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df = processFiles(droneHeight, communicationRadius, minSet,
    #                                                                     generate, addToOriginal, numSensors)

    originalDF = df.copy()
    discreteDF = discretizeData(df)
    signal.signal(signal.SIGINT, signal_handler)
    pd.set_option('display.max_rows', 30)
    pd.set_option('display.max_columns', None)
    # print(discreteDF)
    # print(originalDF)
    if exhaustive:
        # x, y, line, ax2, fig2 = createMSEPlot()
        x, y1, y2, line1, line2, ax2, ax3, fig2 = createIGMSEPlot()
    else:
        x, y1, y2, line1, line2, ax2, ax3, fig2 = createIGMSEPlot()
        IGToSHP = {}

    print(f"Total number of Hoverpoints: {len(HP_gdf)}")
    print(f"Total number of sensors: {len(getSensorNames(HP_gdf['geometry'], sensorNames))}")
    print(f"EPSILON-GREEDY ALGORITHM WITH {numLoops} LOOPS AND ENERGY BUDGET {energyBudget}")
    arProb = 0  # probability of adding (0) and removing (1)
    rbProb = 1  # probability of random (1) and best (0)
    loopCount = 0
    pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond,
                                       dataSize, transferRate)

    minMSE = 1000
    maxIG = float('-inf')
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
            if not exhaustive:
                df = discreteDF
            UHP_gdf, SHP_gdf = addBestHP(unselected=UHP_gdf, unselectedIB=pointsInBudget, selected=SHP_gdf,
                                         sensorNames=sensorNames, df=df, energyWeight=energyWeight,
                                         exhaustive=exhaustive)
        else:
            if not exhaustive:
                df = discreteDF
            UHP_gdf, SHP_gdf = remBestHP(UHP_gdf, SHP_gdf, sensorNames=sensorNames, df=df, energyWeight=energyWeight,
                                         joulesPerMeter=joulesPerMeter, joulesPerSecond=joulesPerSecond,
                                         dataSize=dataSize, transferRate=transferRate, exhaustive=exhaustive)

        UHP_gdf = UHP_gdf.reset_index(drop=True)
        SHP_gdf = SHP_gdf.reset_index(drop=True)
        rbProb = rbProb - 1 / numLoops

        features = getSensorNames(SHP_gdf['geometry'], sensorNames)
        if exhaustive:
            mse = getMSE(features, originalDF)
            if mse < minMSE:
                minMSE = mse
                bestSHP = SHP_gdf.copy()
                iterationOfBest = loopCount
        else:
            # mse = getMSE(features, originalDF)
            mse = 1
            if mse < minMSE:
                minMSE = mse

        IG = getInformationGain(features, discreteDF)
        if IG > maxIG:    # and len(SHP_gdf) > 8:
            maxIG = IG
            if not exhaustive:
                bestSHP = SHP_gdf.copy()
                iterationOfBest = loopCount
        # if not exhaustive:
        #     if IG not in IGToSHP.keys():
        #         IGToSHP[IG] = SHP_gdf.copy()
        if SHP_gdf.empty:
            totalEnergy = 0
        else:
            SHP_gdf, _ = getEnergy(dataSize=dataSize, joulesPerMeter=joulesPerMeter, joulesPerSecond=joulesPerSecond,
                                   selected=SHP_gdf, sensorNames=sensorNames, transferRate=transferRate)
            totalEnergy = SHP_gdf['energy'][0]

        print(f"This set of hoverpoints requires {totalEnergy} Joules out of the"
              f" {energyBudget} Joules in the drone's battery")
        print('Total Number of Hover Points Visited: ', len(SHP_gdf))
        print(f"Total number of sensors visited: {len(getSensorNames(SHP_gdf['geometry'], sensorNames))}")
        if exhaustive:
            print(f"current mse: {mse}, lowest mse yet: {minMSE}")
            # updateMSEPlot(newX=loopCount, newY=mse, ax=ax2, fig=fig2, x=x, y=y, line=line)
            updateIGMSEPlot(newX=loopCount, newY1=mse, newY2=IG,
                            line1=line1, line2=line2, ax2=ax2, ax3=ax3,
                            fig=fig2, x=x, y1=y1, y2=y2)
        else:
            print(f"current mse: {mse}, lowest mse yet: {minMSE}")
            updateIGMSEPlot(newX=loopCount, newY1=mse, newY2=IG, line1=line1, line2=line2, ax2=ax2, ax3=ax3,
                            fig=fig2, x=x, y1=y1, y2=y2)
        print(f"current Information Gain: {IG}, largest IG yet: {maxIG}")
        pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond,
                                           dataSize, transferRate)

        if len(UHP_gdf) == 0:
            arProb = totalEnergy / energyBudget
        else:
            arProb = (totalEnergy / energyBudget) * (
                    1 - (len(pointsInBudget) / len(UHP_gdf)))  # as we approach budget, more likely to remove
        if len(UHP_gdf) + len(SHP_gdf) != len(HP_gdf):
            print('ERROR: SELECTED + UNSELECTED != HP')
            print(len(SHP_gdf), '+', len(UHP_gdf), ' != ', len(HP_gdf))
            exit(1)
    if not exhaustive:
        # print('Checking MSE of 10 hoverpoint sets that had the lowest entropy\n')
        # iterationOfBest = 0
        # tenLowestEntropys = sorted(entropyToSHP.keys())[:20]  # RIGHT NOW IS CHECKING 20 LOWEST ENTROPIES
        # minMSE = 999
        # for entropy in tenLowestEntropys:
        #     selected = entropyToSHP[entropy]
        #     features = getSensorNames(selected['geometry'], sensorNames)
        #     mse = getMSE(features, originalDF)
        #     print(f'mse: {mse}, previous minMSE: {minMSE}')
        #     if mse < minMSE:
        #         minMSE = mse
        #         bestSHP = selected.copy()
        features = getSensorNames(bestSHP.geometry, sensorNames)
        minMSE = getMSE(features, originalDF)


    else:
        features = getSensorNames(bestSHP.geometry, sensorNames)
        minMSE = getMSE(features, originalDF)
    features = getSensorNames(SHP_gdf.geometry, sensorNames)
    finalMSE2 = getMSE(features, originalDF)
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
    print(f"EPSILON-GREEDY ALGORITHM WITH {numLoops} LOOPS AND ENERGY WEIGHT {energyWeight}")
    printResults(bestSHP, iterationOfBest, distanceOfBest, minMSE, sensorNames)
    ax2.scatter(iterationOfBest, minMSE, color='red', zorder=10)
    plotPath(ax1, bestSHP)
    bestSHP.plot(ax=ax1, color='red', markersize=10, alpha=1)
    printTime(startTime)
    for key, value in sensorNames.items():
        if len(value) == 1:
            ax1.text(key.x, key.y, str(value[0]), fontsize=10, ha='center', va='bottom')
    if savePlots:
        writeResults(bestSHP, iterationOfBest, distanceOfBest, minMSE, sensorNames, outputTextName, startTime, finalMSE2)
        fig1.savefig(pathPlotName, bbox_inches='tight')
        fig2.savefig(msePlotName, bbox_inches='tight')
    else:
        plt.show()


def epsilonGreedyHybrid(numLoops, startTime, savePlots=False, pathPlotName="", msePlotName="", outputTextName="",
                  droneHeight=15, communicationRadius=70, energyBudget=60000, joulesPerMeter=10,
                  joulesPerSecond=35, dataSize=100, transferRate=9, generate=False, numSensors=37):
    fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df = processFiles(droneHeight, communicationRadius, False,
                                                                        generate, True, numSensors)
    originalDF = df.copy()
    discreteDF = discretizeData(df)
    x, y1, y2, line1, line2, ax2, ax3, fig2 = createIGMSEPlot()

    print(f"Total number of Hoverpoints: {len(HP_gdf)}")
    print(f"Total number of sensors: {len(getSensorNames(HP_gdf['geometry'], sensorNames))}")
    print(f"EPSILON-GREEDY ALGORITHM WITH {numLoops} LOOPS AND ENERGY BUDGET {energyBudget}")
    arProb = 0  # probability of adding (0) and removing (1)
    rbProb = 1  # probability of random (1) and best (0)
    loopCount = 0
    pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond,
                                       dataSize, transferRate)

    minMSE = float('inf')
    minEntropy = float('inf')
    mse = 99
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
            UHP_gdf, SHP_gdf, mse = addBestHybrid(unselected=UHP_gdf, unselectedIB=pointsInBudget, selected=SHP_gdf,
                                                sensorNames=sensorNames, originalDF=originalDF, discreteDF=discreteDF,
                                                     oldMSE=mse)

        else:
            UHP_gdf, SHP_gdf, mse = remBestHybrid(UHP_gdf, SHP_gdf, sensorNames=sensorNames, originalDF=originalDF,
                                discreteDF=discreteDF, joulesPerMeter=joulesPerMeter, joulesPerSecond=joulesPerSecond,
                                         dataSize=dataSize, transferRate=transferRate, oldMSE=mse)

        UHP_gdf = UHP_gdf.reset_index(drop=True)
        SHP_gdf = SHP_gdf.reset_index(drop=True)
        rbProb = rbProb - 1 / numLoops

        features = getSensorNames(SHP_gdf['geometry'], sensorNames)
        if randomNumber < raProb + rrProb:
            mse = getMSE(features, originalDF)
        if mse < minMSE:
            minMSE = mse
            bestSHP = SHP_gdf.copy()
            iterationOfBest = loopCount

        entropy = getConditionalEntropy(features, discreteDF)
        if entropy < minEntropy:    # and len(SHP_gdf) > 8:
            minEntropy = entropy
        if SHP_gdf.empty:
            totalEnergy = 0
        else:
            SHP_gdf, _ = getEnergy(dataSize=dataSize, joulesPerMeter=joulesPerMeter, joulesPerSecond=joulesPerSecond,
                                   selected=SHP_gdf, sensorNames=sensorNames, transferRate=transferRate)
            totalEnergy = SHP_gdf['energy'][0]

        print(f"This set of hoverpoints requires {totalEnergy} Joules out of the"
              f" {energyBudget} Joules in the drone's battery")
        print('Total Number of Hover Points Visited: ', len(SHP_gdf))

        updateIGMSEPlot(newX=loopCount, newY1=mse, newY2=entropy, line1=line1, line2=line2, ax2=ax2, ax3=ax3,
                            fig=fig2, x=x, y1=y1, y2=y2)
        print(f"current entropy: {entropy}, smallest entropy yet: {minEntropy}")
        print(f'current mse: {mse}, smallest mse yet: {minMSE}')
        pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond,
                                           dataSize, transferRate)

        if len(UHP_gdf) == 0:
            arProb = totalEnergy / energyBudget
        else:
            arProb = (totalEnergy / energyBudget) * (
                    1 - (len(pointsInBudget) / len(UHP_gdf)))  # as we approach budget, more likely to remove
        if len(UHP_gdf) + len(SHP_gdf) != len(HP_gdf):
            print('ERROR: SELECTED + UNSELECTED != HP')
            print(len(SHP_gdf), '+', len(UHP_gdf), ' != ', len(HP_gdf))
            exit(1)
    # features = getSensorNames(bestSHP.geometry, sensorNames)
    # minMSE = getMSE(features, originalDF)
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
    print(f"EPSILON-GREEDY ALGORITHM WITH {numLoops} LOOPS AND ENERGY BUDGET {energyBudget}")
    printResults(bestSHP, iterationOfBest, distanceOfBest, minMSE, sensorNames)
    ax2.scatter(iterationOfBest, minMSE, color='red', zorder=10)
    plotPath(ax1, bestSHP)
    bestSHP.plot(ax=ax1, color='red', markersize=10, alpha=1)
    printTime(startTime)
    for key, value in sensorNames.items():
        if len(value) == 1:
            ax1.text(key.x, key.y, str(value[0]), fontsize=10, ha='center', va='bottom')
    if savePlots:
        writeResults(bestSHP, iterationOfBest, distanceOfBest, minMSE, sensorNames, outputTextName, startTime)
        fig1.savefig(pathPlotName, bbox_inches='tight')
        fig2.savefig(msePlotName, bbox_inches='tight')
    else:
        plt.show()
