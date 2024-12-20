import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.patches as mpatches


# DONE
def realBudgetMSE():
    EG_x = [20,
            25,
            30,
            35,
            40,
            45,
            50,
            55
            ]
    EG_mse = [5.29747E-05,
              2.55E-05,
              1.69666E-05,
              1.24362E-05,
              6.46562E-06,
              2.63358E-06,
              1.30633E-06,
              0]
    EG_error = [3.46256E-06,
                0.00E+00,
                4.66554E-06,
                2.5587E-06,
                3.07777E-06,
                3.20528E-07,
                3.05964E-07,
                0
                ]
    EGIG_mse = [9.30626E-05,
                5.99E-05,
                5.22768E-05,
                3.87828E-05,
                2.72276E-05,
                1.23429E-05,
                1.35766E-05,
                0
                ]
    EGIG_error = [3.94253E-06,
                  3.13E-05,
                  1.80246E-05,
                  8.7783E-06,
                  3.46185E-06,
                  4.55321E-06,
                  1.52854E-05,
                  0
                  ]
    RSEO_x = [30,
              35,
              40,
              45,
              50,
              55
              ]
    RSEO_mse = [0.000178001,
                6.4195E-05,
                3.59503E-05,
                3.38373E-05,
                2.25668E-05,
                0
                ]
    RSEO_error = [6.80905E-05,
                  1.76772E-05,
                  4.43183E-06,
                  1.71806E-05,
                  7.68043E-06,
                  0
                  ]
    Greedy_mse = [8.97E-05,
                  4.52E-05,
                  4.47E-05,
                  4.06E-05,
                  3.34E-05,
                  3.38E-05,
                  1.21E-05,
                  0.00E+00
                  ]
    Greedy_error = [9.03E-07,
                    1.56E-06,
                    3.63E-06,
                    7.67E-07,
                    6.05E-07,
                    1.14E-06,
                    1.31E-06,
                    0.00E+00
                    ]

    EG_mse = [x * 1e5 for x in EG_mse]
    EG_error = [x * 1e5 for x in EG_error]
    EGIG_mse = [x * 1e5 for x in EGIG_mse]
    EGIG_error = [x * 1e5 for x in EGIG_error]
    RSEO_mse = [x * 1e5 for x in RSEO_mse]
    RSEO_error = [x * 1e5 for x in RSEO_error]
    Greedy_mse = [x * 1e5 for x in Greedy_mse]
    Greedy_error = [x * 1e5 for x in Greedy_error]

    plt.figure()
    plt.rcParams.update(
        {'font.size': 25, 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large', 'axes.titlesize': 'x-large'})
    plt.grid(True)
    xticks = [20, 30, 40, 50]
    plt.xticks(xticks)
    yticks = [0, 5, 10, 15, 20]
    plt.yticks(yticks)
    plt.errorbar(RSEO_x, RSEO_mse, yerr=RSEO_error, label='RSEO', linestyle='-', marker='o', markersize=12, capsize=5)
    plt.errorbar(EG_x, EGIG_mse, yerr=EGIG_error, label='Fast-DRONE', linestyle='-', marker='s', markersize=12,
                 capsize=5)
    plt.errorbar(EG_x, EG_mse, yerr=EG_error, label='DRONE', linestyle='-', marker='^', markersize=12, capsize=5)
    plt.errorbar(EG_x, Greedy_mse, yerr=Greedy_error, label='Greedy', linestyle='-', marker='d', markersize=12,
                 capsize=5)
    plt.xlabel(r'Energy Budget (x$10^3$ J)')
    plt.ylabel(r'MSE (x$10^{-5}$)')
    plt.legend(fontsize=20)
    plt.savefig('New Plots/realBudgetMSE', bbox_inches='tight')
    # plt.show()
    plt.close()


# realBudgetMSE()

# DONE
def realBudgetSensors():
    EG_x = [20,
            25,
            30,
            35,
            40,
            45,
            50,
            55
            ]
    EG_num = [3.6,
              6.00E+00,
              9.8,
              15.4,
              20,
              26.8,
              32.6,
              37
              ]
    EG_error = [0.894427191,
                0.00E+00,
                1.095445115,
                3.847076812,
                2.645751311,
                1.303840481,
                1.140175425,
                0

                ]
    EGIG_num = [7.4,
                9.40E+00,
                14.8,
                19.8,
                24.4,
                31.2,
                34,
                37,
                ]
    EGIG_error = [1.341640786,
                  8.94E-01,
                  2.683281573,
                  1.303840481,
                  0.547722558,
                  0.836660027,
                  0.707106781,
                  0

                  ]
    RSEO_x = [20, 25, 30,
              35,
              40,
              45,
              50,
              55
              ]
    RSEO_num = [0, 0, 5.80E+00,
                12,
                20,
                26.2,
                32,
                37
                ]
    RSEO_error = [0, 0, 1.788854382,
                  1,
                  2.738612788,
                  0.836660027,
                  2.549509757,
                  0
                  ]
    Greedy_num = [6,
                  11,
                  16,
                  20,
                  22,
                  27,
                  35,
                  37
                  ]
    Greedy_error = [0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                    ]
    EG_num = [x / 37 * 100 for x in EG_num]
    EGIG_num = [x / 37 * 100 for x in EGIG_num]
    RSEO_num = [x / 37 * 100 for x in RSEO_num]
    EG_error = [x / 37 * 100 for x in EG_error]
    EGIG_error = [x / 37 * 100 for x in EGIG_error]
    RSEO_error = [x / 37 * 100 for x in RSEO_error]
    Greedy_num = [x / 37 * 100 for x in Greedy_num]

    plt.figure()
    plt.rcParams.update(
        {'font.size': 25, 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large', 'axes.titlesize': 'x-large'})
    plt.grid(True)
    xticks = [20, 30, 40, 50]
    plt.xticks(xticks)
    yticks = [0, 25, 50, 75, 100]
    plt.yticks(yticks)
    plt.errorbar(RSEO_x, RSEO_num, yerr=RSEO_error, label='RSEO', linestyle='-', marker='o', markersize=12, capsize=5)
    plt.errorbar(EG_x, EGIG_num, yerr=EGIG_error, label='Fast-DRONE', linestyle='-', marker='s', markersize=12,
                 capsize=5)
    plt.errorbar(EG_x, EG_num, yerr=EG_error, label='DRONE', linestyle='-', marker='^', markersize=12, capsize=5)
    plt.errorbar(EG_x, Greedy_num, yerr=Greedy_error, label='Greedy', linestyle='-', marker='d', markersize=12,
                 capsize=5)
    plt.xlabel(r'Energy Budget (x$10^3$ J)')
    plt.ylabel('% of Visited Sensors')
    plt.legend(fontsize=20)
    plt.savefig('New Plots/realBudgetSensors', bbox_inches='tight')
    plt.show()
    plt.close()


# realBudgetSensors()


def realTechMSE():
    categories = ['Zigbee', 'WiFi', 'BT', 'UWB']
    RSEO_mse = [9.44379E-05,
                4.06E-05,
                4.41299E-05,
                3.90669E-05,
                ]
    EGIG_mse = [4.29094E-05,
                8.24E-06,
                2.42502E-05,
                2.56405E-05,
                ]
    EG_mse = [1.89293E-05,
              2.88E-06,
              4.46604E-06,
              4.26118E-06,
              ]
    EG_err = [3.85259E-06,
              8.96E-07,
              2.01393E-07,
              8.49974E-07,
              ]
    RSEO_err = [0,
                1.04E-05,
                3.54519E-06,
                6.56338E-06,
                ]
    EGIG_err = [0,
                4.41E-06,
                5.56113E-06,
                6.99745E-06,
                ]
    Greedy_mse = [4.38E-05,
                  3.24E-06,
                  2.57E-05,
                  2.08E-05]

    plt.figure()
    plt.rcParams.update(
        {'font.size': 25, 'xtick.labelsize': 30, 'ytick.labelsize': 'x-large',
         'axes.titlesize': 'x-large', 'hatch.linewidth': 2})
    RSEO_mse = [x * 10 ** 6 for x in RSEO_mse]
    EGIG_mse = [x * 10 ** 6 for x in EGIG_mse]
    EG_mse = [x * 10 ** 6 for x in EG_mse]
    EG_err = [x * 10 ** 6 for x in EG_err]
    RSEO_err = [x * 10 ** 6 for x in RSEO_err]
    EGIG_err = [x * 10 ** 6 for x in EGIG_err]
    Greedy_mse = [x * 10 ** 6 for x in Greedy_mse]

    # plt.grid(True)
    barWidth = 0.2
    r1 = np.arange(len(RSEO_mse))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    plt.bar(r1, RSEO_mse, yerr=RSEO_err, width=barWidth, edgecolor='grey', label='RSEO', capsize=5)
    plt.bar(r2, EGIG_mse, yerr=EGIG_err, width=barWidth, edgecolor='grey', label='Fast-DRONE', capsize=5)
    plt.bar(r3, Greedy_mse, width=barWidth, edgecolor='grey', label='Greedy', capsize=5, color='C3')
    plt.bar(r4, EG_mse, yerr=EG_err, width=barWidth, edgecolor='grey', label='DRONE', capsize=5, color='C2')
    plt.xlabel('Technology')
    plt.xticks([r + barWidth for r in range(len(RSEO_mse))], categories)
    # for label in plt.gca().get_xticklabels():
    #     if label.get_text() == 'Zigbee':
    #         label.set_fontsize(26)
    plt.yticks([0, 20, 40, 60, 80, 100])
    plt.ylabel(r'MSE (x$10^{-6}$)')
    plt.legend(fontsize=20)
    plt.savefig('New Plots/realTechMSE',
                bbox_inches='tight')
    # plt.show()
    plt.close()


realTechMSE()


# def realTechFlightTimes():
#     # Step 2: Prepare Your Data and Energy Portions
#     categories = ['Zigbee', 'WiFi', 'BT', 'UWB']
#
#     flying_time = np.array([[762, 717, 625],
#                             [1080, 1110, 1110],
#                             [1062, 1110, 1080],
#                             [1130, 1130, 1140]])
#     bottom_data = flying_time
#     hovering_time = np.array([[360, 420, 480], [25.8, 35, 29.2],
#                               [10.2, 13.7, 10.8], [2.53, 3, 3.36]])
#     top_data = hovering_time
#     bar_width = 0.2  # Width of the bars
#     r1 = np.arange(len(categories))  # Positions for the first algorithm
#     r2 = [x + bar_width for x in r1]  # Positions for the second algorithm
#     r3 = [x + bar_width for x in r2]  # Positions for the third algorithm
#     algorithms = ['RSEO', 'Info Gain', 'Exhaustive']
#     colors_bottom = ['C0', 'C1', 'C2']
#     dark_orange = (1, 0.35, 0)
#     colors_top = ['darkblue', dark_orange, 'darkgreen']
#     plt.figure()
#     plt.rcParams.update(
#         {'font.size': 25, 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large',
#          'axes.titlesize': 'x-large'})
#     # Plotting the bars
#     for i in range(len(algorithms)):
#         bottom_data = list(flying_time[:, i])
#         top_data = list(hovering_time[:, i])
#
#         plt.bar(r1, bottom_data, color=colors_bottom[i], width=bar_width, edgecolor='grey')
#         plt.bar(r1, top_data, bottom=bottom_data, color=colors_top[i], width=bar_width, edgecolor='grey',
#                 label='Top' if i == 0 else "")
#
#         r1 = [x + bar_width for x in r1]
#     # plt.grid(True)
#     # Adding labels
#     plt.xlabel('Technology')
#     plt.ylabel('Time (S)')
#     plt.yticks([0, 250, 500, 750, 1000])
#     plt.xticks([r + bar_width for r in range(len(categories))], categories)
#     for label in plt.gca().get_xticklabels():
#         if label.get_text() == 'Zigbee':
#             label.set_fontsize(26)
#     labels = ['RSEO HT', 'RSEO FT', 'DRONE HT', 'DRONE FT',
#               'CALF HT',
#               'CALF FT']
#     allColors = ['darkblue', 'C0', dark_orange, 'C1', 'darkgreen', 'C2']
#     patches = [mpatches.Patch(color=color, label=label) for color, label in zip(allColors, labels)]
#     plt.subplots_adjust(bottom=0.1)
#     # plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=20)
#     plt.legend(handles=patches, fontsize=20)
#     plt.savefig('Experiments/plots/realTechTimes3',
#                 bbox_inches='tight')
#     plt.close()


def realTechFlightTimes():
    # Step 2: Prepare Your Data and Energy Portions
    categories = ['Zigbee', 'WiFi', 'BT', 'UWB']

    flying_time = np.array([[762, 717, 856.8, 625],
                            [1080, 1110, 1121.7, 1110],
                            [1062, 1110, 1131.1, 1080],
                            [1130, 1130, 1139.4, 1140]])
    bottom_data = flying_time
    hovering_time = np.array([[360, 420, 280, 480], [25.8, 35, 21.11, 29.2],
                              [10.2, 13.7, 9.2, 10.8], [2.53, 3, 2.1, 3.36]])
    top_data = hovering_time
    bar_width = 0.2  # Width of the bars
    r1 = np.arange(len(categories))  # Positions for the first algorithm

    # Define bar positions for each algorithm
    positions = [r1 + bar_width * i for i in range(len(flying_time[0]))]

    algorithms = ['RSEO', 'Fast-DRONE', 'Greedy', 'DRONE']
    colors = ['C0', 'C1', 'C3', 'C2']  # Colors for the bars
    hatches = ['//', '//', '//', '//']  # Patterns for the top bars

    plt.figure()
    plt.rcParams.update(
        {'font.size': 25, 'xtick.labelsize': 30, 'ytick.labelsize': 'x-large',
         'axes.titlesize': 'x-large', 'hatch.linewidth': 2})
    # Plotting the bars
    for i in range(len(algorithms)):
        plt.bar(positions[i], flying_time[:, i], color=colors[i], width=bar_width, edgecolor='black',
                label=f'{algorithms[i]} FT')
        plt.bar(positions[i], hovering_time[:, i], bottom=flying_time[:, i], color=colors[i], width=bar_width,
                edgecolor='black', hatch=hatches[i], label=f'{algorithms[i]} HT')

    # Adding labels
    plt.xlabel('Technology')
    plt.ylabel('Time (S)')
    plt.yticks([0, 250, 500, 750, 1000])
    plt.xticks([r + bar_width for r in range(len(categories))], categories)

    # for label in plt.gca().get_xticklabels():
    #     if label.get_text() == 'Zigbee':
    #         label.set_fontsize(42)

    # Create legend handles manually
    # patches = [mpatches.Patch(facecolor=color, hatch=hatch, label=f'{algorithm} HT', edgecolor='black') for color, algorithm, hatch in zip(colors, algorithms, hatches)]
    patches = [mpatches.Patch(color=color, label=f'{algorithm}') for algorithm, color in zip(algorithms, colors)]
    patches.append(mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Hover-time'))
    patches.append(mpatches.Patch(facecolor='white', edgecolor='black', label='Fly-time'))

    # plt.subplots_adjust(bottom=0.1)
    plt.legend(handles=patches, handlelength=1, fontsize=19, loc='lower right')

    # Save the figure
    plt.savefig('New Plots/realTechTimes', bbox_inches='tight')
    # plt.show()
    plt.close()


realTechFlightTimes()


def numSensorsMSE():
    EG_mse = [2.80E-06,
              2.70E-06,
              2.93E-06,
              2.94E-06,
              3.74E-06,
              3.86E-06
              ]
    EG_error = [9.76564E-07,
                2.87777E-07,
                6.97353E-07,
                6.25208E-07,
                7.10403E-07,
                8.94754E-07
                ]
    EGIG_mse = [4.53E-06,
                4.18E-06,
                4.71E-06,
                5.29E-06,
                5.80E-06,
                6.54E-06
                ]
    EGIG_error = [1.62628E-06,
                  9.23542E-07,
                  1.14569E-06,
                  9.68606E-07,
                  2.7044E-06,
                  1.45597E-06
                  ]
    x2 = [20,
          30,
          40,
          50,
          60,
          70
          ]
    RSEO_mse = [9.81E-06,
                1.36E-05,
                1.14E-05,
                1.34E-05,
                1.37E-05,
                1.61E-05
                ]
    RSEO_error = [5.10E-06,
                  2.89222E-06,
                  2.42E-06,
                  3.95738E-06,
                  5.48E-06,
                  8.27177E-06
                  ]
    Greedy_mse = [5.77E-06,
                  4.33E-06,
                  6.10E-06,
                  6.96E-06,
                  7.67E-06,
                  9.80E-06]
    Greedy_error = [3.99368E-06,
                    1.26758E-06,
                    1.63344E-06,
                    2.74104E-06,
                    1.41891E-06,
                    1.51643E-06]

    # relative_difference = (EG_mse[-1] - EG_mse[0]) / EG_mse[0] * 100
    # print(f"Relative difference EG: {relative_difference:.2f}%")
    # relative_difference = (EGIG_mse[-1] - EGIG_mse[0]) / EGIG_mse[0] * 100
    # print(f"Relative difference EGIG: {relative_difference:.2f}%")
    # relative_difference = (RSEO_mse[-1] - RSEO_mse[0]) / RSEO_mse[0] * 100
    # print(f"Relative difference RSEO: {relative_difference:.2f}%")
    # lastTwoRSEO = RSEO_mse  # [-2:]
    # lastTwoEGIG = EGIG_mse  # [-2:]
    # percentage_differences = [(EG - EGIG) / EGIG * 100 for EG, EGIG in zip(lastTwoEGIG, lastTwoRSEO)]
    # print(lastTwoEGIG)
    # print(lastTwoRSEO)
    # print(percentage_differences)
    # print(sum(percentage_differences) / len(percentage_differences))
    # average_percentage_difference = sum(percentage_differences) / len(percentage_differences)
    #
    # print("On average, RSEO is {:.2f}% higher than EGIG.".format(average_percentage_difference))
    # exit()

    EG_mse = [x * 1e5 for x in EG_mse]
    EG_error = [x * 1e5 for x in EG_error]
    EGIG_mse = [x * 1e5 for x in EGIG_mse]
    EGIG_error = [x * 1e5 for x in EGIG_error]
    RSEO_mse = [x * 1e5 for x in RSEO_mse]
    RSEO_error = [x * 1e5 for x in RSEO_error]
    Greedy_mse = [x * 1e5 for x in Greedy_mse]
    Greedy_error = [x * 1e5 for x in Greedy_error]
    plt.figure()
    plt.rcParams.update(
        {'font.size': 25, 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large', 'axes.titlesize': 'x-large'})
    plt.grid(True)
    plt.errorbar(x2, RSEO_mse, yerr=RSEO_error, label='RSEO', linestyle='-', marker='o', markersize=12, capsize=5)
    plt.errorbar(x2, EGIG_mse, yerr=EGIG_error, label='Fast-DRONE', linestyle='-', marker='s', markersize=12, capsize=5)
    plt.errorbar(x2, EG_mse, yerr=EG_error, label='DRONE', linestyle='-', marker='^', markersize=12, capsize=5)
    plt.errorbar(x2, Greedy_mse, yerr=Greedy_error, label='Greedy', linestyle='-', marker='d', markersize=12, capsize=5)
    plt.xlabel('# of Available Sensors')
    plt.ylabel(r'MSE (x$10^{-5}$)')
    plt.yticks([0, .5, 1, 1.5, 2])
    plt.xticks([20, 30, 40, 50, 60, 70])
    plt.legend(fontsize=20)
    plt.savefig('New Plots/numSensorsMSE',
                bbox_inches='tight')
    # plt.show()
    plt.close()


# numSensorsMSE()

def numSensorsSensors():
    EG_num = [1.37E+01,
              1.75E+01,
              1.85E+01,
              2.22E+01,
              2.58E+01,
              2.64E+01
              ]
    EG_error = [1.86E+00,
                1.643167673,
                2.664582519,
                1.834847859,
                1.940790217,
                1.140175425
                ]
    EGIG_num = [1.50E+01,
                1.88E+01,
                2.25E+01,
                2.52E+01,
                2.95E+01,
                3.20E+01

                ]
    EGIG_error = [2.449489743,
                  2.483277404,
                  1.974841766,
                  3.250640962,
                  1.974841766,
                  1.897366596

                  ]
    x2 = [20,
          30,
          40,
          50,
          60,
          70

          ]
    RSEO_num = [1.44E+01,
                1.51E+01,
                1.72E+01,
                1.71E+01,
                2.07E+01,
                1.87E+01

                ]
    RSEO_error = [3.282952601,
                  3.282952601,
                  4.918784854,
                  4.594682917,
                  5.049752469,
                  3.81E+00
                  ]
    Greedy_num = [1.43E+01,
                  1.80E+01,
                  2.00E+01,
                  2.57E+01,
                  3.10E+01,
                  3.43E+01
                  ]
    Greedy_error = [3.511884584,
                    4,
                    1,
                    2,
                    2,
                    2.886751346
                    ]

    EG_num = [sensors / total * 100 for sensors, total in zip(EG_num, x2)]
    EGIG_num = [sensors / total * 100 for sensors, total in zip(EGIG_num, x2)]
    RSEO_num = [sensors / total * 100 for sensors, total in zip(RSEO_num, x2)]
    EG_error = [sensors / total * 100 for sensors, total in zip(EG_error, x2)]
    EGIG_error = [sensors / total * 100 for sensors, total in zip(EGIG_error, x2)]
    RSEO_error = [sensors / total * 100 for sensors, total in zip(RSEO_error, x2)]
    Greedy_sensors = [sensors / total * 100 for sensors, total in zip(Greedy_num, x2)]
    Greedy_error = [sensors / total * 100 for sensors, total in zip(Greedy_error, x2)]

    plt.figure()
    plt.rcParams.update(
        {'font.size': 25, 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large', 'axes.titlesize': 'x-large'})
    plt.grid(True)
    plt.ylim(0, 100)
    plt.xticks([20, 30, 40, 50, 60, 70])
    plt.yticks([0, 25, 50, 75, 100])
    plt.errorbar(x2, RSEO_num, yerr=RSEO_error, label='RSEO', linestyle='-', marker='o', markersize=12, capsize=5)
    plt.errorbar(x2, EGIG_num, yerr=EGIG_error, label='Fast-DRONE', linestyle='-', marker='s', markersize=12, capsize=5)
    plt.errorbar(x2, EG_num, yerr=EG_error, label='DRONE', linestyle='-', marker='^', markersize=12, capsize=5)
    plt.errorbar(x2, Greedy_sensors, yerr=Greedy_error, label='Greedy', linestyle='-', marker='d', markersize=12, capsize=5)
    plt.xlabel('# of Available Sensors')
    plt.ylabel('% of Visited Sensors')
    plt.legend(fontsize=20)
    plt.savefig('New Plots/numSensorsSensors', bbox_inches='tight')
    plt.show()
    plt.close()


numSensorsSensors()


# DONE
def synthBudgetMSE():
    x = [20,
         25,
         30,
         35,
         40,
         45,
         50,
         55
         ]
    EG_mse = [2.54E-05,
              2.46E-05,
              1.98E-05,
              9.52E-06,
              4.64E-06,
              2.67E-06,
              1.17706E-06,
              7.31747E-07
              ]
    EG_error = [1.83117E-07,
                1.35187E-07,
                3.31883E-06,
                1.51939E-07,
                1.61895E-07,
                2.74365E-07,
                8.27071E-08,
                8.8651E-08

                ]
    EGIG_mse = [3.15E-05,
                3.12E-05,
                2.83E-05,
                1.11E-05,
                9.31E-06,
                4.80E-06,
                3.00863E-06,
                1.70692E-06

                ]
    EGIG_error = [3.69332E-06,
                  3.39149E-06,
                  2.75632E-06,
                  1.69885E-07,
                  3.73556E-06,
                  6.42178E-07,
                  1.04386E-06,
                  1.09451E-06

                  ]

    RSEO_mse = [6.85E-05,
                6.85E-05,
                5.21E-05,
                2.44E-05,
                2.13E-05,
                1.97E-05,
                1.78327E-05,
                6.79924E-06
                ]
    RSEO_error = [0,
                  0,
                  8.29919E-21,
                  1.0342E-05,
                  3.00288E-07,
                  2.93563E-06,
                  4.57967E-06,
                  6.07931E-06
                  ]
    Greedy_mse = [3.34E-05,
                  3.13E-05,
                  1.85E-05,
                  1.10E-05,
                  7.02E-06,
                  5.67E-06,
                  5.39E-06,
                  4.40E-06
                  ]

    # Calculate how many times as big RSEO is than EG
    # RSEO_vs_EG = [RSEO / EG for RSEO, EG in zip(RSEO_mse, EG_mse)]
    #
    # # Calculate how many times as big RSEO is than EGIG
    # RSEO_vs_EGIG = [RSEO / EGIG for RSEO, EGIG in zip(RSEO_mse, EGIG_mse)]
    # print(f'rseo / eg = {RSEO_vs_EG}')
    # print(f'rseo / egig = {RSEO_vs_EGIG}')
    # exit()
    # percentage_differences = [(EGIG - EG) / EG * 100 for EG, EGIG in zip(EG_mse, EGIG_mse)]
    # print(EG_mse)
    # print(EGIG_mse)
    # print(percentage_differences)
    # average_percentage_difference = sum(percentage_differences) / len(percentage_differences)
    #
    # print("On average, EGIG is {:.2f}% higher than EG.".format(average_percentage_difference))
    # exit()
    EG_mse = [x * 1e6 for x in EG_mse]
    EG_error = [x * 1e6 for x in EG_error]
    EGIG_mse = [x * 1e6 for x in EGIG_mse]
    EGIG_error = [x * 1e6 for x in EGIG_error]
    RSEO_mse = [x * 1e6 for x in RSEO_mse]
    RSEO_error = [x * 1e6 for x in RSEO_error]
    Greedy_mse = [x * 1e6 for x in Greedy_mse]
    # Greedy_error = [x * 1e6 for x in Greedy_error]
    plt.figure()
    plt.rcParams.update(
        {'font.size': 25, 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large', 'axes.titlesize': 'x-large'})
    plt.grid(True)
    xticks = [20, 30, 40, 50]
    plt.xticks(xticks)
    yticks = [0, 20, 40, 60, 80]
    plt.yticks(yticks)
    plt.errorbar(x, RSEO_mse, yerr=RSEO_error, label='RSEO', linestyle='-', marker='o', markersize=12, capsize=5)
    plt.errorbar(x, EGIG_mse, yerr=EGIG_error, label='Fast-DRONE', linestyle='-', marker='s', markersize=12, capsize=5)
    plt.errorbar(x, EG_mse, yerr=EG_error, label='DRONE', linestyle='-', marker='^', markersize=12, capsize=5)
    plt.errorbar(x, Greedy_mse, label='Greedy', linestyle='-', marker='d', markersize=12, capsize=5)
    plt.xlabel(r'Energy Budget (x$10^3$ J)')
    plt.ylabel(r'MSE (x$10^{-6}$)')
    plt.legend(fontsize=20)
    plt.savefig('New Plots/synthBudgetMSE', bbox_inches='tight')
    # plt.show()
    plt.close()


# synthBudgetMSE()


# DONE
def synthBudgetSensors():
    x = [20,
         25,
         30,
         35,
         40,
         45,
         50,
         55
         ]
    EG_num = [9.00E+00,
              9.00E+00,
              1.00E+01,
              1.70E+01,
              2.27E+01,
              2.93E+01,
              33.66666667,
              37.66666667

              ]
    EG_error = [0,
                0,
                1.732050808,
                2.645751311,
                0.577350269,
                1.527525232,
                0.577350269,
                0.577350269

                ]
    EGIG_num = [1.07E+01,
                1.27E+01,
                1.47E+01,
                1.83E+01,
                2.57E+01,
                3.03E+01,
                35.33333333,
                38.33333333

                ]
    EGIG_error = [1.154700538,
                  2.081665999,
                  0.577350269,
                  0.577350269,
                  1.154700538,
                  1.154700538,
                  0.577350269,
                  0.577350269

                  ]

    RSEO_num = [3.00E+00,
                3.00E+00,
                5,
                1.13E+01,
                2.23E+01,
                2.43E+01,
                30,
                37

                ]
    RSEO_error = [0,
                  0,
                  0.577350269,
                  1.154700538,
                  0.577350269,
                  0.577350269,
                  0,
                  4.358898944
                  ]
    Greedy_num = [9,
                  8,
                  10,
                  18,
                  24,
                  30,
                  34,
                  39
                  ]
    # Greedy_error = [0, 0, 0, 0, 0, 0, 0, 0]

    EG_num = [x / 37 * 100 for x in EG_num]
    EGIG_num = [x / 37 * 100 for x in EGIG_num]
    RSEO_num = [x / 37 * 100 for x in RSEO_num]
    EG_error = [x / 37 * 100 for x in EG_error]
    EGIG_error = [x / 37 * 100 for x in EGIG_error]
    RSEO_error = [x / 37 * 100 for x in RSEO_error]
    Greedy_num = [x / 37 * 100 for x in Greedy_num]

    plt.figure()
    plt.rcParams.update(
        {'font.size': 25, 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large', 'axes.titlesize': 'x-large'})
    plt.grid(True)
    xticks = [20, 30, 40, 50]
    plt.xticks(xticks)
    yticks = [0, 25, 50, 75, 100]
    plt.yticks(yticks)
    plt.errorbar(x, RSEO_num, yerr=RSEO_error, label='RSEO', linestyle='-', marker='o', markersize=12, capsize=5)
    plt.errorbar(x, EGIG_num, yerr=EGIG_error, label='Fast-DRONE', linestyle='-', marker='s', markersize=12, capsize=5)
    plt.errorbar(x, EG_num, yerr=EG_error, label='DRONE', linestyle='-', marker='^', markersize=12, capsize=5)
    plt.errorbar(x, Greedy_num, label='Greedy', linestyle='-', marker='d', markersize=12, capsize=5)
    plt.xlabel(r'Energy Budget (x$10^3$ J)')
    plt.ylabel('% of Visited Sensors')
    plt.legend(fontsize=20)
    plt.savefig('New Plots/synthBudgetSensors', bbox_inches='tight')
    # plt.show()
    plt.close()

# synthBudgetSensors()

# EG_times = [40, 120, 236, (510 + 422 + 460)/3, (772 + 648 + 810)/3]
# # EG_times = [(1015 + 1064 + 1266 + 1604) / 4]
# print(EG_times)
# EGIG_times = [15, 20, 28, 46, 60]
# # EGIG_times = [(84 + 92 + 123 + 82)/4]
# percent_improvements = [
#     ((eg - egig) / eg) * 100 for eg, egig in zip(EG_times, EGIG_times)
# ]
# #
# # Print out the percent improvements
# for i, percent_improvement in enumerate(percent_improvements):
#     print(f"Test {i + 1}: {percent_improvement:.2f}% improvement")
# synthBudgetSensors()
#
# # synthBudgetSensors()
# # def techHoveringPoints():
# #     categories = ['Zigbee', 'WiFi', 'BT', 'UWB']
# #     RSEO_num = [6,
# #                 15.5,
# #                 18,
# #                 18.5,
# #                 ]
# #     EGIG_num = [7,
# #                 21,
# #                 23,
# #                 22.5,
# #                 ]
# #     EG_num = [8,
# #               19,
# #               20,
# #               22.5,
# #               ]
# #     # EG_err = [3.85259E-06,
# #     #           8.96E-07,
# #     #           2.01393E-07,
# #     #           8.49974E-07,
# #     #           ]
# #     # RSEO_err = [0,
# #     #             1.04E-05,
# #     #             3.54519E-06,
# #     #             6.56338E-06,
# #     #             ]
# #     # EGIG_err = [0,
# #     #             4.41E-06,
# #     #             5.56113E-06,
# #     #             6.99745E-06,
# #     #             ]
# #     plt.figure()
# #     plt.rcParams.update(
# #         {'font.size': 25, 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large',
# #          'axes.titlesize': 'x-large'})
# #     # RSEO_mse = [x * 10 ** 5 for x in RSEO_mse]
# #     # EGIG_mse = [x * 10 ** 5 for x in EGIG_mse]
# #     # EG_mse = [x * 10 ** 5 for x in EG_mse]
# #     # EG_err = [x * 10 ** 5 for x in EG_err]
# #     # RSEO_err = [x * 10 ** 5 for x in RSEO_err]
# #     # EGIG_err = [x * 10 ** 5 for x in EGIG_err]
# #
# #     # plt.grid(True)
# #     barWidth = 0.25
# #     r1 = np.arange(len(RSEO_num))
# #     r2 = [x + barWidth for x in r1]
# #     r3 = [x + barWidth for x in r2]
# #     plt.bar(r1, RSEO_num, width=barWidth, edgecolor='grey', label='RSEO', capsize=5)
# #     plt.bar(r2, EGIG_num, width=barWidth, edgecolor='grey', label='DRONE', capsize=5)
# #     plt.bar(r3, EG_num, width=barWidth, edgecolor='grey', label='CALF', capsize=5)
# #     plt.xlabel('Technology')
# #     plt.xticks([r + barWidth for r in range(len(RSEO_num))], categories)
# #     # plt.yticks([0, 2, 4, 6, 8, 10])
# #     plt.ylabel('# of Hovering Points')
# #     plt.legend(fontsize=20)
# #     plt.savefig('Experiments/plots/realTechHP',
# #                 bbox_inches='tight')
# #     plt.close()
#
# # techHoveringPoints()
#
# # def techDistance():
# #     categories = ['Zigbee', 'WiFi', 'BT', 'UWB']
# #     RSEO_num = [3.09E+03,
# #                 3790,
# #                 3800,
# #                 3950
# #                 ]
# #     EGIG_num = [3000,
# #                 3900,
# #                 3890,
# #                 3960,
# #                 ]
# #     EG_num = [2750,
# #               3910,
# #               3800,
# #               3990,
# #               ]
# #     # EG_err = [3.85259E-06,
# #     #           8.96E-07,
# #     #           2.01393E-07,
# #     #           8.49974E-07,
# #     #           ]
# #     # RSEO_err = [0,
# #     #             1.04E-05,
# #     #             3.54519E-06,
# #     #             6.56338E-06,
# #     #             ]
# #     # EGIG_err = [0,
# #     #             4.41E-06,
# #     #             5.56113E-06,
# #     #             6.99745E-06,
# #     #             ]
# #     plt.figure()
# #     plt.rcParams.update(
# #         {'font.size': 25, 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large',
# #          'axes.titlesize': 'x-large'})
# #     # RSEO_mse = [x * 10 ** 5 for x in RSEO_mse]
# #     # EGIG_mse = [x * 10 ** 5 for x in EGIG_mse]
# #     # EG_mse = [x * 10 ** 5 for x in EG_mse]
# #     # EG_err = [x * 10 ** 5 for x in EG_err]
# #     # RSEO_err = [x * 10 ** 5 for x in RSEO_err]
# #     # EGIG_err = [x * 10 ** 5 for x in EGIG_err]
# #
# #     # plt.grid(True)
# #     barWidth = 0.25
# #     r1 = np.arange(len(RSEO_num))
# #     r2 = [x + barWidth for x in r1]
# #     r3 = [x + barWidth for x in r2]
# #     plt.bar(r1, RSEO_num, width=barWidth, edgecolor='grey', label='RSEO', capsize=5)
# #     plt.bar(r2, EGIG_num, width=barWidth, edgecolor='grey', label='DRONE', capsize=5)
# #     plt.bar(r3, EG_num, width=barWidth, edgecolor='grey', label='CALF', capsize=5)
# #     plt.xlabel('Technology')
# #     plt.xticks([r + barWidth for r in range(len(RSEO_num))], categories)
# #     # plt.yticks([0, 2, 4, 6, 8, 10])
# #     plt.ylabel('Distance Traveled (m)')
# #     plt.legend(fontsize=20, loc='lower right')
# #     plt.savefig('Experiments/plots/realTechDistance',
# #                 bbox_inches='tight')
# #     plt.close()
# #
# # # techDistance()
# #
# # def techHoveringPoints2():
# #     # Step 2: Prepare Your Data and Energy Portions
# #     categories = ['Zigbee', 'WiFi', 'BT', 'UWB']
# #     RSEO_hp = [6,
# #                 15.5,
# #                 18,
# #                 18.5,
# #                 ]
# #     EGIG_hp = [7,
# #                 21,
# #                 23,
# #                 22.5,
# #                 ]
# #     EG_hp = [8,
# #               19,
# #               20,
# #               22.5,
# #               ]
# #     numSensors = np.array([[12, 12, 10.4],
# #                             [27.8, 33.4, 27],
# #                             [17.6, 21.8, 19.2],
# #                             [18.2, 22.2, 21]])
# #     numHover = np.array([[6, 7, 8], [15.5, 21, 19],
# #                          [17.6, 21.8, 19.2],
# #                          [18.2, 22.2, 21]])
# #     numSensors = numSensors - numHover
# #     bar_width = 0.2  # Width of the bars
# #     r1 = np.arange(len(categories))  # Positions for the first algorithm
# #     r2 = [x + bar_width for x in r1]  # Positions for the second algorithm
# #     r3 = [x + bar_width for x in r2]  # Positions for the third algorithm
# #     algorithms = ['RSEO', 'Info Gain', 'Exhaustive']
# #     colors_bottom = ['C0', 'C1', 'C2']
# #     dark_orange = (1, 0.35, 0)
# #     colors_top = ['darkblue', dark_orange, 'darkgreen']
# #     plt.figure()
# #     plt.rcParams.update(
# #         {'font.size': 25, 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large',
# #          'axes.titlesize': 'x-large'})
# #     # Plotting the bars
# #     for i in range(len(algorithms)):
# #         bottom_data = list(numHover[:, i])
# #         top_data = list(numSensors[:, i])
# #
# #         plt.bar(r1, bottom_data, color=colors_bottom[i], width=bar_width, edgecolor='grey')
# #         plt.bar(r1, top_data, bottom=bottom_data, color=colors_top[i], width=bar_width, edgecolor='grey',
# #                 label='Top' if i == 0 else "")
# #
# #         r1 = [x + bar_width for x in r1]
# #     # plt.grid(True)
# #     # Adding labels
# #     plt.xlabel('Technology')
# #     plt.ylabel('# of Sensors/HPs')
# #     plt.yticks([0, 10, 20, 30])
# #     plt.xticks([r + bar_width for r in range(len(categories))], categories)
# #     for label in plt.gca().get_xticklabels():
# #         if label.get_text() == 'Zigbee':
# #             label.set_fontsize(26)
# #     labels = ['RSEO sens', 'RSEO hov', 'DRONE sens', 'DRONE hov',
# #               'CALF sens',
# #               'CALF hov']
# #     allColors = ['darkblue', 'C0', dark_orange, 'C1', 'darkgreen', 'C2']
# #     patches = [mpatches.Patch(color=color, label=label) for color, label in zip(allColors, labels)]
# #     plt.subplots_adjust(bottom=0.1)
# #     plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=20)
# #     # plt.legend(handles=patches, fontsize=20)
# #     plt.savefig('Experiments/plots/techHPSensors',
# #                 bbox_inches='tight')
# #     plt.close()
# #
# # # techHoveringPoints2()
