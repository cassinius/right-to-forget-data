TARGETS = ["education", "marital", "income"]

import os, csv
import numpy as np
import matplotlib.pyplot as plt

k_factors = ['5', '10', '20', '50', '100', '200']

ORIGINAL_SCORE = {
  'education': {
    'linear_svc': 0.638625,
    'logistic_regression': 0.579927,
    'gradient_boosting': 0.639540,
    'random_forest': 0.596712
  },
  'marital': {
    'linear_svc': 0.852719,
    'logistic_regression': 0.800801,
    'gradient_boosting': 0.846007,
    'random_forest': 0.829944
  },
  'income': {
    'linear_svc': 0.830889,
    'logistic_regression': 0.792534,
    'gradient_boosting': 0.832195,
    'random_forest': 0.801783
  }
}

RESULTS_DIR = "../../iml_output/"
FIGURES_DIR = RESULTS_DIR + "figures/"

ALGORITHMS = ["linear_svc", "logistic_regression", "random_forest", "gradient_boosting"]

markers = ['o', '^', 'D', 'x', 'v', 'p', 'H']
linestyles = ['-', '--', '-.', '-.-', '.-.', ':']
colors = ['g', 'saddlebrown', 'r', 'b', 'm', 'k', 'g']

equals = {
  5: [], 10: [], 20: [], 50: [], 100: [], 200: []
}
biass = {
  5: [], 10: [], 20: [], 50: [], 100: [], 200: []
}
imls = {
  5: [], 10: [], 20: [], 50: [], 100: [], 200: []
}

# for target in TARGETS:

target = "income"

for algo_str in ALGORITHMS:
  results_file_list = [f for f in sorted(os.listdir(RESULTS_DIR)) if f.startswith("results") and f.endswith(".csv")]
  for file in results_file_list:
    file_arr = file.split(".")[0].split("_")
    # print(file_arr)
    if file_arr[len(file_arr)-1] != target or file_arr[len(file_arr)-2] != algo_str.split("_")[1]:
      continue

    # print(file_arr)
    results_file = open(RESULTS_DIR + file, 'r')
    results_csv = csv.reader(results_file, delimiter=',')
    next(results_csv, None)

    for line in results_csv:
      # print(line)

      if line[0] == 'equal':
        equals[int(line[1])].append(float(line[2]))
      if line[0] == 'bias':
        biass[int(line[1])].append(float(line[2]))
      if line[0] == 'iml':
        imls[int(line[1])].append(float(line[2]))

    equal = [ORIGINAL_SCORE[target][algo_str], np.mean(equals[5]), np.mean(equals[10]), np.mean(equals[20]), np.mean(equals[50]), np.mean(equals[100]), np.mean(equals[200])]
    bias = [ORIGINAL_SCORE[target][algo_str], np.mean(biass[5]), np.mean(biass[10]), np.mean(biass[20]), np.mean(biass[50]), np.mean(biass[100]), np.mean(biass[200])]
    iml = [ORIGINAL_SCORE[target][algo_str], np.mean(imls[5]), np.mean(imls[10]), np.mean(imls[20]), np.mean(imls[50]), np.mean(imls[100]), np.mean(imls[200])]

    print(equal)
    print(bias)
    print(iml)

    min_score = min(min(equal), min(bias), min(iml))
    max_score = max(max(equal), max(bias), max(iml))

    print("Min score: %s" %(min_score))
    print("Max score: %s" %(max_score))

    x = range(0, len(k_factors) + 1)
    x_labels = ['none'] + k_factors

    fig, ax = plt.subplots()
    rect = fig.patch
    rect.set_facecolor('white')

    plt.title("F1 score dependent on k-factor, %s, %s" % (target, algo_str))

    equal_line = plt.plot(equal, marker='o', linestyle='-', color='r', label="equal weights")
    bias_line, = plt.plot(bias, marker='^', linestyle='-', color='b', label="human bias")
    iml_line, = plt.plot(iml, marker='D', linestyle='-', color='g', label="human iML")

    # Create a legend (Matplotlib madness...!!!)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    plt.axis([0, 6, float(min_score), float(max_score)])
    plt.xticks(x, x_labels)
    plt.xlabel('anonymization k-factor')
    plt.ylabel('F1 score')
    # plt.show()

    plt.savefig(FIGURES_DIR + target + "_" + algo_str + ".png")