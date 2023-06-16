# DroneFeatureSelection
Project Description: There are sensors spread throughout a field and in order to collect their data you must travel to them, so a drone is used. The amount of sensors the drone can visit is limited by the battery of the drone. Machine Learning models can be used to predict the values of the unvisited sensors from the values of the ones visited by the drone. Our goal is to maximize the accuracy of these predictions by selecting the best subset of features for the drone to travel to without exceeding the energy budget of the drone's battery.

mapping.py: Processes the shp file to plot all of the sensor locations, their communication radius and hover points. Hover points are the midpoints of all overlapping sections of communication radius'. For now, if a hover point is in a section shared by two communication radius', then traveling there means you have to recieve from both sensors, and the same is true for sections shared by three sensors. Also contains the code to plot the path of the drone.

ML.py: Contains the code to create our dataset and determine the most accurate machine learning model. Based on the results of the testModels function we defined, a Gradient Boosting Regressor was chosen as the model for the main algorithm.

TSP.py: Contains hoverPointsTSP function, which runs a traveling salesman heuristic on the set of points given and returns a geoDataFrame with these ordered points in the geometry column.

main.py: Contains the main epsilon-greedy program to add/remove random or the best hoverpoints to the current selected-hover-points set with varying probabilities within a while loop. As the program approaches its end it is more likely to add/remove the best hoverpoints, which is computed by the distance each hoverpoint would add/remove to the path of the drone and the mean-squared-error it would add to training and testing a gradient boosting regressor when the sensors corresponding to that hoverpoint are added/removed from the set of selected sensors. The program starts by adding hover-points, and as the number of unselected hover-points that are in budget decreases, the probability of removing increases. When the while loop is complete, it plots the path of the drone and the mse of each loop iteration.
