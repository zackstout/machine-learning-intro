
# following sentdex
# (https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/?completed=/label-data-machine-learning/):

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm


# SVC = support vector classifier (ex. of latter)
# SVM = support vector machine
# features (X) = features we want to classify (axes of graph)
# labels (y) = what each dot represents (e.g. a dog or a cat)

# note: svm.SVC(kernel='linear', C = 1.0).fit(X, y) is deprecated.
# you have to use cross_validation_train.train_test_split on your X and y, and use KNeighborsClassifier() before we can .fit()

# normalization of data: put all features on a similar scale.

# it's a part of "pre-processing" your data before ML.

# tradeoff between accuracy (how many did it classify right) and performance (what's the EV, basically)

# Quandl: grab the package to simplify things. Gets you a ton of datasets.

# syntax diffs between 2.7 and 3?


# linear regression: m = (mean(x)*mean(y) - mean(x*y)) / ((mean(x))^2 - mean(x^2))
# And b = mean(y) - m*mean(x)
# and what do you know, mean is just an np method!

# you should write that parentheses-checker now.

# regression_line = [(m*x)+b for x in xs]
# plt.scatter(xs, ys)
# plt.plot(xs, regression_line)
# plt.show()

# allows you to predict y when x is a given input.

# how good/accurate/confident (the latter two stay similar for lower numbers of features/axes) is our best-fit line?

# one way of answering:
# coefficient of determination (r^2): square the error to penalize outliers. also gets rid of negative values. (well, at least where |e| < 1 ....
# r^2 = 1 - (SE(y(bestfit)) / SE(mean(y)))
# We want r^2 to be close to 1 (that indicates linearity of data)

# hmm, when you import random, it's only "pseudo-random" ...!























# hi
