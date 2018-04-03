
# Following https://www.datacamp.com/community/tutorials/machine-learning-python:


from sklearn import datasets
import pandas as pd

digits = datasets.load_digits()

print(digits)

# Could also do it like this:
# digits = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)


print(digits.keys())

print(digits.data)

print(digits.target)

print(digits.DESCR)












# chillin
