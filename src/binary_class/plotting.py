import csv
import matplotlib.pyplot as plt


# MODE = 'anonymization'
MODE = 'perturbation'


# TARGET = 'education_num'
TARGET = 'marital_status'


# Input files
ALGORITHMS = {
  'gradient_boost': '../../output/' + MODE + '/adults_target_' + TARGET + '/results_gradient_boosting.csv',
  'logistic_regression': '../../output/' + MODE + '/adults_target_' + TARGET + '/results_logistic_regression.csv',
  'onevsrest_bagging': '../../output/' + MODE + '/adults_target_' + TARGET + '/results_onevsrest_bagging.csv',
  'random_forest': '../../output/' + MODE + '/adults_target_' + TARGET + '/results_random_forest.csv',
  'linear_svc': '../../output/' + MODE + '/adults_target_' + TARGET + '/results_linear_svc.csv'
}
ALGO = ALGORITHMS['gradient_boost']


# Perturbation Dataset names
marital_status_perturbation_files = [
  'relationship_Husband',
  'relationship_Not-in-family',
  'relationship_Unmarried',
  'sex_Female',
  'sex_Male'
]


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



def plotAnonymizationResults(results):
  k_factors = ['03', '07', '11', '15', '19', '23', '27', '31', '35', '100']
  equal_line_f1 = [results["adults_original_dataset.csv"]["f1"], ]
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

  fig = plt.figure()
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

  # print len(results)
  # print perturbation_files

  lines = {}

  markers = ['o', '^', 'D', 'x', 'v', 'p' ]
  linestyles = ['-', '--', '-.', '-.-', '.-.', ':']
  colors = ['r', 'g', 'b', 'c', 'm', 'k']

  min_score = float('inf')
  max_score = float('-inf')


  for file in perturbation_files:
    line = [
      results["original_dataset.csv"]["f1"]
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
    plt.plot(line, marker=markers[idx], color=colors[idx], label=key)

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
  results = readResultsIntoHash(ALGO)
  if MODE == 'anonymization':
    plotAnonymizationResults(results)
  elif MODE == 'perturbation':
    plotPerturbationResults(results, marital_status_perturbation_files)
  else:
    print "This mode is not supported."
