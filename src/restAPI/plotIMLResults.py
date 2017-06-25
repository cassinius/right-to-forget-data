import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

IMG_DIR= "/var/www/iMLAnonResultPlots/"

if "IML_SERVER" in os.environ:
    print( "Found IML Server environment entry, saving plots to: " + os.environ["IML_SERVER"] )
    PLOT_BASE_URL = "http://" + os.environ["IML_SERVER"] + "/iMLAnonResultPlots/"
else:
    print( "Found no IML Server environment entry, saving plots to: localhost. " )
    PLOT_BASE_URL = "http://localhost/iMLAnonResultPlots/"



def plotAndWriteResultsToFS(overall_results):
    print( "Plotting..." )

    iml = overall_results['results']['iml']
    bias = overall_results['results']['bias']

    N = 4
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    bias_results = (bias['gradient_boosting']['f1'],
                    bias['linear_svc']['f1'],
                    bias['logistic_regression']['f1'],
                    bias['random_forest']['f1'])

    iml_results = (iml['gradient_boosting']['f1'],
                   iml['linear_svc']['f1'],
                   iml['logistic_regression']['f1'],
                   iml['random_forest']['f1'])

    min_score = min(min(bias_results), min(iml_results))
    max_score = max(max(bias_results), max(iml_results))
    min_height = max(0, float(min_score) / 2)
    max_height = min(1, float(max_score) * 1.25)

    print( "Setting MIN height to: " + str(min_height) )
    print( "Setting MAX height to: " + str(max_height) )

    fig, ax = plt.subplots()
    bias_rects = ax.bar(ind, bias_results, width, color='#8b0333')
    iml_rects = ax.bar(ind + width, iml_results, width, color='#78b6db')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('F1 Score')
    ax.set_title('Bias vs. iML Anonymization performance (F1 Score)')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('Gradient Boost', 'Linear SVC', 'Log.Regression', 'Random Forest'))
    ax.legend((bias_rects[0], iml_rects[0]), ('Bias', 'iML'))
    plt.axis([-0.25, 3.6, min_height, max_height])

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%.2f' % float(height),
                    ha='center', va='bottom')

    autolabel(bias_rects)
    autolabel(iml_rects)

    filename = overall_results['grouptoken'] + "_" + overall_results['usertoken'] + "_" + overall_results[
        'target'] + "_" + overall_results['timestamp'] + '.png'

    overall_results['plotURL'] = PLOT_BASE_URL + filename

    fig.savefig(IMG_DIR + filename, bbox_inches='tight')
    # plt.show()



if __name__ == "__main__":
  overall_results = {
      "usertoken": "testuser",
      "grouptoken": "testgroup",
      "target": "marital-status",
      "timestamp": '20170526233416',
      "plotURL": '/var/www/iMLAnonResultPlots/testgroup_testuser_marital-status_20170526233416.png',
      "results": {
          "bias": {
              "gradient_boosting": {
                  "accuracy": 0,
                  "f1": 0.72,
                  "precision": 0,
                  "recall": 0
              },
              "linear_svc": {
                  "accuracy": 0,
                  "f1": 0.70,
                  "precision": 0,
                  "recall": 0
              },
              "logistic_regression": {
                  "accuracy": 0,
                  "f1": 0.68,
                  "precision": 0,
                  "recall": 0
              },
              "random_forest": {
                  "accuracy": 0,
                  "f1": 0.75,
                  "precision": 0,
                  "recall": 0
              }
          },
          "iml": {
              "gradient_boosting": {
                  "accuracy": 0,
                  "f1": 0.75,
                  "precision": 0,
                  "recall": 0
              },
              "linear_svc": {
                  "accuracy": 0,
                  "f1": 0.77,
                  "precision": 0,
                  "recall": 0
              },
              "logistic_regression": {
                  "accuracy": 0,
                  "f1": 0.72,
                  "precision": 0,
                  "recall": 0
              },
              "random_forest": {
                  "accuracy": 0,
                  "f1": 0.78,
                  "precision": 0,
                  "recall": 0
              }
          }
      }
  }
  plotAndWriteResultsToFS(overall_results)
