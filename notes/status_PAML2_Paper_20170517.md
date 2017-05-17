# Status PAML2 Paper


## Scenarios

* dataset = adult dataset (US cencus data of 1994)
* 4 different classifier categories used:
    - Logistic regression (theta-vector optimization)
    - LinearSVC (SVM with linear kernel)
    - Random Forest (bagging => shifting towards bias by overfitting on random subsamples & averaging)
    - Gradient Boosting (boosting => building a strong classifier out of many weak ones)
* **MC** = Multi-class
* all anonymization done with k-factors of k = [3, 7, 11, 15, 19, 23, 27, 31, 35, 100]
* binary classification: target column = 'income'
* MC classification: targets = ['education_num', 'marital_status']
    - in case of 'education_num': target grouped into 4 levels representing ['<= ground school', '<= high school diploma', '<= bachelor', '> bachelor']
* rows containing the most significant attributes (according to logit coefficients) removed
* Outlier analysis with removal of percentiles p = [0.1 : 0.95 : 0.05] (5% step-size)


### 1. MC classification of anonymized datasets

* behavior corresponds mainly to binary classification, only the random forest shows surprising behavior by dropping sharply in performance with weak anonymization, only to gain in performance again with increasing k factor...
    - must have something to do with the bagging strategy (random subsampling from the whole population)


### 2. MC classification of perturbed datasets

* harder in MC than binary classification as logit coefs may be very different amongst classes in OVR (one-vs-rest) setup
* this leads to significantly different behaviors between *education_num* and *marital_status* classification targets, as the former displays very different logit coefs for the different classes while the latter displays mostly the same (in-)significant influences across classes
    - removing most (=information) & least (=noise) significant rows shows clean improvement / degradation of performance for marital_status
    - but shows undirected behavor for education_num
    - for both targets, this is consistent across all classifiers


### 3. Binary classification on data with outliers removed

* Performance on data with less variance drops significantly towards the high end of the spectrum (95% of data reduced)
* So it seems that reducing variance (making the data more homogeneous) makes it more difficult for classifiers to decide on a decision boundary
    - surprisingly, performance drops off much more drastically vs. the original dataset than with the anonymized versions...
    - but that might have to do with binary vs. MC classification

#### 3.a comparison to same classification with random data points removed

* results stay practically the same up until 95% of data removed
* no great surprise, since random removal does not change variance / data topology significantly


### 4. MC classification of anonymized datasets (only 'marital_status') based on 30% outliers removed

* reduced variance
* computational time increases by a significant factor (~5-10 times, depends on algorithm, need to measure exactly)
* results are approximately the same as 'normal' anonymization, depending on classifier used
* surprisingly, results here do not drop off as sharply as with variance reduced in a non-anonymized fashion (as seen above), but that might have to do with binary-vs-MC classification
* results with LinearSVC shows same (mysterious??) behavior than RF on 'normal' MC on anonymized data....
    - now I am sure it MUST have something to do with variance in combination with subsampling for RF => those things are connected!!! but how???


### 5. MC on anonymized datasets with most insignificant columns removed???

* not tried yet...
* Expectation: better results than original anonymization
    - maybe sufficiently good so that one could perform ML on anonymized versions with such data removed as a preprocessing step?


## Conclusions

1. More new questions than answers
2. Suprising behavior for Random Forest in MC classification
3. Surprising behavior of classifiers with outliers removed
    3.a. especially since random deletion produces no drop-off
4. Surprisingly, LinearSVC classifier on anonymized data with 30% outliers removed start to show similar behavior to RF on normal anonym. MC (drop-off and subsequent gradual increase in F1 score, although not as )