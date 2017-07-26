TARGETS = ["education", "marital", "income"]

import os, csv
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "../../iml_output/"

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

for target in TARGETS:
  for algo_str in ALGORITHMS:
    results_file_list = [f for f in sorted(os.listdir(RESULTS_DIR)) if f.startswith("results") and f.endswith(".csv")]
    for file in results_file_list:
      file_arr = file.split(".")[0].split("_")
      # print(file_arr)
      if file_arr[len(file_arr)-1] != target or file_arr[len(file_arr)-2] != algo_str.split("_")[1]:
        continue

      print(file_arr)
      results_file = open(RESULTS_DIR + file, 'r')
      results_csv = csv.reader(results_file, delimiter=',')
      next(results_csv, None)

      for line in results_csv:
        print(line)

        