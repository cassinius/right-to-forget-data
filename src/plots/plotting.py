import os, csv
import numpy as np
import matplotlib.pyplot as plt
from src.plots.plots_blur import gradient_fill
from matplotlib import gridspec


# MODE = 'anonymization'
# MODE = 'perturbation'
MODE = 'outliers'


# OUTLIER_TARGET = ''
# OUTLIER_TARGET = 'outliers/'
OUTLIER_TARGET = 'random_comparison/'
# OUTLIER_TARGET = 'original/'
# OUTLIER_TARGET = 'outliers_removed/'

# OUTLIER_PREFIX = 'adults_outliers_removed_'
OUTLIER_PREFIX = 'adults_random_deletion_'

# TARGET = 'education_num/'
# TARGET = 'marital_status/'
TARGET = 'income/'


# Input files
ALGORITHMS = {
  'gradient_boost': '../../output/' + MODE + '/adults_target_' + TARGET + OUTLIER_TARGET + '/results_gradient_boosting.csv',
  'logistic_regression': '../../output/' + MODE + '/adults_target_' + TARGET + OUTLIER_TARGET + '/results_logistic_regression.csv',
  'onevsrest_bagging': '../../output/' + MODE + '/adults_target_' + TARGET + OUTLIER_TARGET + '/results_onevsrest_bagging.csv',
  'random_forest': '../../output/' + MODE + '/adults_target_' + TARGET + OUTLIER_TARGET + '/results_random_forest.csv',
  'linear_svc': '../../output/' + MODE + '/adults_target_' + TARGET + OUTLIER_TARGET + '/results_linear_svc.csv'
}
ALGO = ALGORITHMS['gradient_boost']

OUTLIERS_DIRECTORY = '../../output/outliers/adults_target_' + TARGET + '/' + OUTLIER_TARGET
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
colors = ['g', 'saddlebrown', 'r', 'b', 'm', 'k', 'g']


def readOutlierResultsIntoHash():
  results_file_list = [f for f in sorted(os.listdir(OUTLIERS_DIRECTORY)) if f.startswith("results") and f.endswith(".csv")]
  results = {}
  for input_file in results_file_list:

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

  ### Collect Classification Results ###
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
  print( "Min score: " + min_score )
  print( "Max score: " + max_score )

  x = range(0, len(out_factors) + 1)
  x_labels = [0]
  for o in out_factors:
    x_labels.append(o)
  print( "x: " + str(x))
  print( "Labels: " + str( x_labels ) )


  ### Collect Std. Deviation from data_stats
  ### HACK - refactor out into own function !!!
  std_devs = []
  with open(OUTLIERS_DIRECTORY + "/data_stats.csv", 'r') as f:
    next(f)
    stat_lines = [line.split(',') for line in f]
    for idx, line in enumerate(stat_lines):
      std_devs.append(line[2])
      # print( "line{0} = {1}".format(idx, line) )

  print( std_devs )
  min_std = min(std_devs)
  max_std = max(std_devs)
  print( "Min Std: " + min_std )
  print( "Max Std: " + max_std )

  ### START PLOTTING ###
  fig, (ax_top, ax_bottom) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
  fig.patch.set_facecolor('white')
  # ax.set_axis_bgcolor((116/256.0, 139/256.0, 197/256.0))
  # ax.set_axis_bgcolor((255/256.0, 199/256.0, 0/256.0))
  # ax.set_axis_bgcolor((50/256.0, 50/256.0, 50/256.0))

  if (OUTLIER_TARGET == 'outliers/'):
    target_label = 'outliers'
  else:
    target_label = 'random data points'

  plt.suptitle("F1 score dependent on " + target_label + " removed")

  for idx, key in enumerate(lines):
    line = lines[key]
    gradient_fill( np.array(x),
                   np.array(list(map(float, line))),
                   y_min=float(min_score),
                   y_max=float(max_score),
                   # zfunc=plots_blur.zfunc,
                   ax=ax_top,
                   marker=markers[idx],
                   color=colors[idx],
                   label=key )
    # ax_top.plot(line, marker=markers[idx], color=colors[idx], label=key)

  # Create a legend (Matplotlib madness...!!!)
  handles, labels = ax_top.get_legend_handles_labels()
  ax_top.legend(handles, labels)

  ax_top.axis([0, max(x), float(min_score), float(max_score)])
  ax_top.locator_params(nbins=18, axis='x')
  ax_top.set_xticklabels(x_labels)
  ax_top.set_xlabel('% of ' + target_label + ' removed')
  ax_top.set_ylabel('F1 score')

  gradient_fill(np.array(x),
                np.array(list(map(float, std_devs))),
                y_min=float(min_std),
                y_max=float(max_std),
                # zfunc=plots_blur.zfunc,
                ax=ax_bottom,
                marker=markers[idx],
                color='cyan',
                label='standard deviation')
  # ax_bottom.plot(std_devs)

  ax_bottom.axis([0, max(x), 20000, 45000])
  ax_bottom.locator_params(nbins=18, axis='x')
  ax_bottom.set_xticklabels(x_labels)
  ax_bottom.set_xlabel('% of ' + target_label + ' removed')
  ax_bottom.set_ylabel('Std.Dev.')

  # plt.tight_layout()
  plt.show()




def plotAnonymizationResults(results, original=None):
  if OUTLIER_TARGET == 'outliers_removed/':
    original_dataset = "adults___0.3.csv"
  elif original is None:
    original_dataset = "adults_original_dataset.csv"

  k_factors = ['3', '7', '11', '15', '19', '23', '27', '31', '35', '100']
  equal_line_f1 = [results[original_dataset]["f1"]]
  age_line_f1 = [results[original_dataset]["f1"]]
  race_line_f1 = [results[original_dataset]["f1"]]
  for k in k_factors:
    equal_line_f1.append(results["adults_anonymized_k" + k + "_equal.csv"]["f1"])
    age_line_f1.append(results["adults_anonymized_k" + k + "_emph_age.csv"]["f1"])
    race_line_f1.append(results["adults_anonymized_k" + k + "_emph_race.csv"]["f1"])

  print( "Equal: " + str(equal_line_f1) )
  print( "Emph age: " + str(age_line_f1) )
  print( "Emph race: " + str(race_line_f1) )

  min_score = min(min(equal_line_f1), min(age_line_f1), min(race_line_f1))
  max_score = max(max(equal_line_f1), max(age_line_f1), max(race_line_f1))

  print( "Min score: " + min_score )
  print( "Max score: " + max_score )

  x = range(0, len(k_factors) + 1)
  labels = ['none'] + k_factors

  print( "Labels: " + str( labels ) )

  fig, ax = plt.subplots()
  rect = fig.patch
  rect.set_facecolor('white')

  plt.title("F1 score dependent on k-factor, %s" % (getAlgorithmName()) )

  equal_line, = plt.plot(equal_line_f1, marker='o', linestyle='-', color='r', label="equal weights")
  age_line, = plt.plot(age_line_f1, marker='^', linestyle='-', color='b', label="age preferred")
  race_line, = plt.plot(race_line_f1, marker='D', linestyle='-', color='g', label="race preferred")
  plt.legend(handles=[equal_line, age_line, race_line])

  print( equal_line )
  print( age_line )
  print( race_line )

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
  print( "Max F1 Score: " + str(max_score) )
  print( "Min F1 Score: " + str(min_score) )

  x = [0, 1, 2, 3, 4, 5]
  x_labels = ["0", "20%", "40%", "60%", "80%", "100%"]

  fig, ax = plt.subplots()
  rect = fig.patch
  rect.set_facecolor('white')

  plt.title("F1 score dependent on perturbation, " + getAlgorithmName())

  for idx, key in enumerate(lines):
    line = lines[key]
    gradient_fill(np.array(x),
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
    print( "Mode %s This mode is not supported." % (MODE) )
