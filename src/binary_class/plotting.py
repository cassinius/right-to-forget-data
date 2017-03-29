import csv
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Adults income files
# gradient_boost_file = '../output/results_gradient_boosting.csv'
# logistic_regression_file = '../output/results_logistic_regression.csv'
# onevsrest_bagging_file = '../output/results_onevsrest_bagging.csv'
# random_forest_file = '../output/results_random_forest.csv'
# svm_linear_file = '../output/results_svm_linear.csv'


# Adults education-num files
# gradient_boost_file = '../../output/adults_target_education_num/results_gradient_boosting.csv'
# logistic_regression_file = '../../output/adults_target_education_num/results_logistic_regression.csv'
# onevsrest_bagging_file = '../../output/adults_target_education_num/results_onevsrest_bagging.csv'
# random_forest_file = '../../output/adults_target_education_num/results_random_forest.csv'
# linear_svc_file = '../../output/adults_target_education_num/results_linear_svc.csv'


# Adults marital-status files
gradient_boost_file = '../../output/adults_target_marital_status/results_gradient_boosting.csv'
logistic_regression_file = '../../output/adults_target_marital_status/results_logistic_regression.csv'
onevsrest_bagging_file = '../../output/adults_target_marital_status/results_onevsrest_bagging.csv'
random_forest_file = '../../output/adults_target_marital_status/results_random_forest.csv'
linear_svc_file = '../../output/adults_target_marital_status/results_linear_svc.csv'



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

  title_algo = "Random Forest"
  plt.title("F1 score dependent on k-factor, %s" % (title_algo) )

  equal_line, = plt.plot(equal_line_f1, marker='o', linestyle='-', color='r', label="equal weights")
  age_line, = plt.plot(age_line_f1, marker='^', linestyle='-', color='b', label="age preferred")
  race_line, = plt.plot(race_line_f1, marker='D', linestyle='-', color='g', label="race preferred")
  plt.legend(handles=[equal_line, age_line, race_line])

  plt.axis([0, 5, float(min_score), float(max_score)])
  plt.xticks(x, labels)
  plt.xlabel('anonymization k-factor')
  plt.ylabel('F1 score')
  plt.show()


def plotPerturbationResultsTop3(results):
  capital_gain = [
    results["adults_original_dataset.csv"]["f1"],
    results["adults_capital-gain_2000_0.2.csv"]["f1"],
    results["adults_capital-gain_2000_0.4.csv"]["f1"],
    results["adults_capital-gain_2000_0.6.csv"]["f1"],
    results["adults_capital-gain_2000_0.8.csv"]["f1"],
    results["adults_capital-gain_2000_1.csv"]["f1"],
  ]
  education_num = [
    results["adults_original_dataset.csv"]["f1"],
    results["adults_education-num_10_0.2.csv"]["f1"],
    results["adults_education-num_10_0.4.csv"]["f1"],
    results["adults_education-num_10_0.6.csv"]["f1"],
    results["adults_education-num_10_0.8.csv"]["f1"],
    results["adults_education-num_10_1.csv"]["f1"]
  ]
  marital_status = [
    results["adults_original_dataset.csv"]["f1"],
    results["adults_marital-status_Married-civ-spouse_0.2.csv"]["f1"],
    results["adults_marital-status_Married-civ-spouse_0.4.csv"]["f1"],
    results["adults_marital-status_Married-civ-spouse_0.6.csv"]["f1"],
    results["adults_marital-status_Married-civ-spouse_0.8.csv"]["f1"],
    results["adults_marital-status_Married-civ-spouse_1.csv"]["f1"]
  ]

  print capital_gain
  print education_num
  print marital_status

  x = [0, 1, 2, 3, 4, 5]
  labels = ["0", "20%", "40%", "60%", "80%", "100%"]

  fig = plt.figure()
  rect = fig.patch
  rect.set_facecolor('white')

  plt.title("F1 score dependent on perturbation, linear SVC")

  capital_line, = plt.plot(capital_gain, marker='o', linestyle='-', color='r', label="Capital gain > 2k")
  edunum_line, = plt.plot(education_num, marker='^', linestyle='-', color='b', label="Education num > 10")
  marital_line, = plt.plot(marital_status, marker='D', linestyle='-', color='g', label="Marital status, civ spouse")
  plt.legend(handles=[capital_line, edunum_line, marital_line], loc=3)

  plt.axis([0, 5, 0.39, 0.67])
  plt.xticks(x, labels)
  plt.xlabel('degree of perturbation')
  plt.ylabel('F1 score')
  plt.show()


def plotPerturbationResultsBottom3(results):
  marital_status = [
    results["adults_original_dataset.csv"]["f1"],
    results["adults_marital-status_Never-married_0.2.csv"]["f1"],
    results["adults_marital-status_Never-married_0.4.csv"]["f1"],
    results["adults_marital-status_Never-married_0.6.csv"]["f1"],
    results["adults_marital-status_Never-married_0.8.csv"]["f1"],
    results["adults_marital-status_Never-married_1.csv"]["f1"]
  ]
  occupation = [
    results["adults_original_dataset.csv"]["f1"],
    results["adults_occupation_Other-service_0.2.csv"]["f1"],
    results["adults_occupation_Other-service_0.4.csv"]["f1"],
    results["adults_occupation_Other-service_0.6.csv"]["f1"],
    results["adults_occupation_Other-service_0.8.csv"]["f1"],
    results["adults_occupation_Other-service_1.csv"]["f1"]
  ]
  relationship = [
    results["adults_original_dataset.csv"]["f1"],
    results["adults_relationship_Own-child_0.2.csv"]["f1"],
    results["adults_relationship_Own-child_0.4.csv"]["f1"],
    results["adults_relationship_Own-child_0.6.csv"]["f1"],
    results["adults_relationship_Own-child_0.8.csv"]["f1"],
    results["adults_relationship_Own-child_1.csv"]["f1"]
  ]

  print marital_status
  print occupation
  print relationship

  x = [0, 1, 2, 3, 4, 5]
  labels = ["0", "20%", "40%", "60%", "80%", "100%"]

  fig = plt.figure()
  rect = fig.patch
  rect.set_facecolor('white')

  plt.title("F1 score dependent on perturbation, gradient boost")

  capital_line, = plt.plot(marital_status, marker='o', linestyle='-', color='r', label="Never married")
  edunum_line, = plt.plot(occupation, marker='^', linestyle='-', color='b', label="Other service job")
  marital_line, = plt.plot(relationship, marker='D', linestyle='-', color='g', label="Own child")
  plt.legend(handles=[capital_line, edunum_line, marital_line], loc=2)

  plt.axis([0, 5, 0.70, 0.72])
  plt.xticks(x, labels)
  plt.xlabel('degree of perturbation')
  plt.ylabel('F1 score')
  plt.show()


if __name__ == "__main__":
  algorithm = "random_forest"
  target = "education_num"

  results = readResultsIntoHash(random_forest_file)
  plotAnonymizationResults(results)
  # plotPerturbationResultsTop3(results)
  # plotPerturbationResultsBottom3(results)
