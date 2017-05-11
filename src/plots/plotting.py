import os, csv
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import matplotlib.patches as patches
# from PIL import Image
# from PIL import ImageDraw
# from PIL import ImageFilter
# np.random.seed(1979)
import plots_blur


# MODE = 'anonymization'
# MODE = 'perturbation'
MODE = 'outliers'

# OUTLIER_MODE = 'outliers'
OUTLIER_MODE = 'random_comparison'

# OUTLIER_PREFIX = 'adults_outliers_removed_'
OUTLIER_PREFIX = 'adults___'

# TARGET = 'education_num'
# TARGET = 'marital_status'
TARGET = 'income'


# Input files
ALGORITHMS = {
  'gradient_boost': '../../output/' + MODE + '/adults_target_' + TARGET + '/results_gradient_boosting.csv',
  'logistic_regression': '../../output/' + MODE + '/adults_target_' + TARGET + '/results_logistic_regression.csv',
  'onevsrest_bagging': '../../output/' + MODE + '/adults_target_' + TARGET + '/results_onevsrest_bagging.csv',
  'random_forest': '../../output/' + MODE + '/adults_target_' + TARGET + '/results_random_forest.csv',
  'linear_svc': '../../output/' + MODE + '/adults_target_' + TARGET + '/results_linear_svc.csv'
}
ALGO = ALGORITHMS['random_forest']

OUTLIERS_DIRECTORY = '../../output/outliers/adults_target_' + TARGET + '/' + OUTLIER_MODE
OUTLIERS_ALGORITHMS = ['gradient_boosting', 'logistic_regression', 'random_forest', 'linear_svc']

PERTURBATION_FILES = {
  'education_num': [
    'age_0',
    'marital-status_Divorced',
    'marital-status_Married-civ-spouse',
    'occupation_Tech-support',
    'relationship_Husband',
    'workclass_Federal-gov'
  ],
  'marital_status': [
    'age_0',
    'relationship_Husband',
    'relationship_Not-in-family',
    'relationship_Unmarried',
    'relationship_Own-child',
    'relationship_Wife'
  ]
}

markers = ['o', '^', 'D', 'x', 'v', 'p', 'H']
linestyles = ['-', '--', '-.', '-.-', '.-.', ':']
colors = ['r', 'y', 'b', 'c', 'm', 'k', 'g']


def readOutlierResultsIntoHash():
  print OUTLIERS_DIRECTORY
  filelist = [f for f in sorted(os.listdir(OUTLIERS_DIRECTORY)) if f.endswith(".csv")]
  results = {}
  for input_file in filelist:

    # print input_file
    results[input_file] = {}
    results_file = open(OUTLIERS_DIRECTORY + "/" + input_file, 'r')
    results_csv = csv.reader(results_file, delimiter=',')

    # ignore the headers
    next(results_csv, None)

    for line in results_csv:
      results[input_file][line[0]] = {}
      results[input_file][line[0]]['precision'] = line[1]
      results[input_file][line[0]]['recall'] = line[2]
      results[input_file][line[0]]['f1'] = line[3]

    # print results[input_file]
    results_file.close()
  return results


def readResultsIntoHash(file_name):
  results_file = open(file_name, 'r')
  results_csv = csv.reader(results_file, delimiter=',')

  # ignore the headers
  next(results_csv, None)

  # create the dict we need
  results = {}

  for line in results_csv:
    results[line[0]] = {}
    results[line[0]]['precision'] = line[1]
    results[line[0]]['recall'] = line[2]
    results[line[0]]['f1'] = line[3]

  results_file.close()
  return results



def plotOutlierResults(results):
  lines = {}
  out_factors = np.linspace(0.1, 0.95, 18)

  linear_svc_line_f1 = [results["results_linear_svc.csv"]["adults_original_dataset.csv"]["f1"]]
  logistic_regression_line_f1 = [results["results_logistic_regression.csv"]["adults_original_dataset.csv"]["f1"]]
  random_forest_line_f1 = [results["results_random_forest.csv"]["adults_original_dataset.csv"]["f1"]]
  gradient_boosting_line_f1 = [results["results_gradient_boosting.csv"]["adults_original_dataset.csv"]["f1"]]

  lines["Linear SVC"] = linear_svc_line_f1
  lines["Logistic Regression"] = logistic_regression_line_f1
  lines["Random Forest"] = random_forest_line_f1
  lines["Gradient Boosting"] = gradient_boosting_line_f1

  for o in out_factors:
    linear_svc_line_f1.append(results["results_linear_svc.csv"][OUTLIER_PREFIX + str(o) + ".csv"]["f1"])
    logistic_regression_line_f1.append(results["results_logistic_regression.csv"][OUTLIER_PREFIX + str(o) + ".csv"]["f1"])
    random_forest_line_f1.append(results["results_random_forest.csv"][OUTLIER_PREFIX + str(o) + ".csv"]["f1"])
    gradient_boosting_line_f1.append(results["results_gradient_boosting.csv"][OUTLIER_PREFIX + str(o) + ".csv"]["f1"])

  min_score = min(min(linear_svc_line_f1), min(logistic_regression_line_f1), min(random_forest_line_f1), min(gradient_boosting_line_f1))
  max_score = max(max(linear_svc_line_f1), max(logistic_regression_line_f1), max(random_forest_line_f1), max(gradient_boosting_line_f1))

  print "Min score: " + min_score
  print "Max score: " + max_score

  x = range(0, len(out_factors) + 1)
  x_labels = ['none']
  for o in out_factors:
    x_labels.append(str(o))
  print "Labels: " + str( x_labels )

  fig, ax = plt.subplots()
  fig.patch.set_facecolor('white')
  # ax.set_axis_bgcolor((116/256.0, 139/256.0, 197/256.0))
  ax.set_axis_bgcolor((255/256.0, 199/256.0, 0/256.0))
  # ax.set_axis_bgcolor((50/256.0, 50/256.0, 50/256.0))

  if (OUTLIER_MODE == 'outliers'):
    target_label = 'outliers'
  else:
    target_label = 'random data points'
  plt.title("F1 score dependent on " + target_label + " removed")

  for idx, key in enumerate(lines):
    line = lines[key]
    plots_blur.gradient_fill(np.array(x),
                             np.array(map(float, line)),
                             y_min=float(min_score),
                             y_max=float(max_score),
                             # zfunc=plots_blur.zfunc,
                             ax=ax,
                             marker=markers[idx],
                             color=colors[idx],
                             label=key)
    # plt.plot(line, marker=markers[idx], color=colors[idx], label=key)

  # Create a legend (Matplotlib madness...!!!)
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels)

  plt.axis([0, 5, float(min_score), float(max_score)])
  plt.xticks(x, x_labels)
  plt.xlabel('degree of ' + target_label + ' removed')
  plt.ylabel('F1 score')
  plt.show()




def plotAnonymizationResults(results):
  k_factors = ['03', '07', '11', '15', '19', '23', '27', '31', '35', '100']
  equal_line_f1 = [results["adults_original_dataset.csv"]["f1"]]
  age_line_f1 = [results["adults_original_dataset.csv"]["f1"]]
  race_line_f1 = [results["adults_original_dataset.csv"]["f1"]]
  for k in k_factors:
    equal_line_f1.append(results["adults_anonymized_k" + k + "_equal.csv"]["f1"])
    age_line_f1.append(results["adults_anonymized_k" + k + "_emph_age.csv"]["f1"])
    race_line_f1.append(results["adults_anonymized_k" + k + "_emph_race.csv"]["f1"])

  print "Equal: " + str(equal_line_f1)
  print "Emph age: " + str(age_line_f1)
  print "Emph race: " + str(race_line_f1)

  min_score = min(min(equal_line_f1), min(age_line_f1), min(race_line_f1))
  max_score = max(max(equal_line_f1), max(age_line_f1), max(race_line_f1))

  print "Min score: " + min_score
  print "Max score: " + max_score

  x = range(0, len(k_factors) + 1)
  labels = ['none'] + k_factors

  print "Labels: " + str( labels )

  fig, ax = plt.subplots()
  rect = fig.patch
  rect.set_facecolor('white')

  plt.title("F1 score dependent on k-factor, %s" % (getAlgorithmName()) )

  equal_line, = plt.plot(equal_line_f1, marker='o', linestyle='-', color='r', label="equal weights")
  age_line, = plt.plot(age_line_f1, marker='^', linestyle='-', color='b', label="age preferred")
  race_line, = plt.plot(race_line_f1, marker='D', linestyle='-', color='g', label="race preferred")
  plt.legend(handles=[equal_line, age_line, race_line])

  print equal_line
  print age_line
  print race_line

  plt.axis([0, 5, float(min_score), float(max_score)])
  plt.xticks(x, labels)
  plt.xlabel('anonymization k-factor')
  plt.ylabel('F1 score')
  plt.show()




'''
  We are always assuming:
  1. an original dataset result
  2. selectively deleted data ranging in amount from
     20 - 100% of the data => so 6 data points per line
'''
def plotPerturbationResults(results, perturbation_files):
  lines = {}

  min_score = float('inf')
  max_score = float('-inf')


  for file in perturbation_files:
    line = [
      results["adults_original_dataset.csv"]["f1"]
    ]
    for fraction in ['0.2', '0.4', '0.6', '0.8', '1']:
      entry = "adults_" + file + "_" + fraction + ".csv"
      # print entry

      f1 = float( results[entry]["f1"] )
      min_score = f1 if f1 < min_score else min_score
      max_score = f1 if f1 > max_score else max_score

      line.append(f1)
    # print line
    lines[file] = line

  # print lines
  print "Max F1 Score: " + str(max_score)
  print "Min F1 Score: " + str(min_score)

  x = [0, 1, 2, 3, 4, 5]
  x_labels = ["0", "20%", "40%", "60%", "80%", "100%"]

  fig, ax = plt.subplots()
  rect = fig.patch
  rect.set_facecolor('white')

  plt.title("F1 score dependent on perturbation, " + getAlgorithmName())

  for idx, key in enumerate(lines):
    line = lines[key]
    plots_blur.gradient_fill(np.array(x),
                             np.array(map(float, line)),
                             y_min=float(min_score),
                             y_max=float(max_score),
                             zfunc=None, #plots_blur.zfunc,
                             ax=ax,
                             marker=markers[idx],
                             color=colors[idx],
                             label=key)
    # plt.plot(line, marker=markers[idx], color=colors[idx], label=key)

  # Create a legend (Matplotlib madness...!!!)
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels)

  plt.axis([0, 5, min_score, max_score])
  plt.xticks(x, x_labels)
  plt.xlabel('degree of perturbation')
  plt.ylabel('F1 score')
  plt.show()


def getAlgorithmName():
  title_algo = ALGO.split('/')
  title_algo = title_algo[len(title_algo)-1]
  title_algo = title_algo.split('.')[0]
  title_algo = title_algo.split('_')[1:]
  title_algo = ' '.join(title_algo).title()
  # title_algo[1] = "SVC" if title_algo[1] == "Svc" else title_algo[1]
  return title_algo


if __name__ == "__main__":
  if MODE == 'outliers':
    results = readOutlierResultsIntoHash()
    plotOutlierResults(results)
  elif MODE == 'anonymization':
    results = readResultsIntoHash(ALGO)
    plotAnonymizationResults(results)
  elif MODE == 'perturbation':
    results = readResultsIntoHash(ALGO)
    plotPerturbationResults(results, PERTURBATION_FILES[TARGET])
  else:
    print "This mode is not supported."
