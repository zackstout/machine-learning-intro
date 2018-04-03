
import numpy as np

# Input (4 rows x 3 cols):
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
# Output (4 rows x 1 col, with T as transpose):
y = np.array([[0,1,1,0]]).T

# Example just uses (3, 1)...:
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
for j in xrange(60000):
    # Using the sigmoid function:
    l1 = 1/(1+np.exp(-(np.dot(X,syn0)))) # what is X? is it j? No! It's the input array!
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)


# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        # A nice property of the sigmoid function:
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
# We have 3 inputs and 1 output (X and y):
syn0 = 2*np.random.random((3,1)) - 1

# go through training set a lot of times to optimize its predictions:
for iter in xrange(10000):

    # forward propagation (does this want to be iter (or syn0?) rather than X...?) No!! It's X, the input array!
    # Our data:
    l0 = X # has size 3 (there are 4 training examples)
    # (4x3 dot 3x1) = guess (length 4, just like examples), then passed through the sigmoid function.
    l1 = nonlin(np.dot(l0,syn0)) # has size 1

    # how much did we miss? (check against original array of length 4)
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    # This is the secret key:
    # Get slope of sigmoid at values in l1.
    # Then, l1_delta is essentially a dot product (multiplying elementwise):
    # The idea is that we multiply high-confidence decisions by a low number.
    # Derivative of sigmoid is low at values near 1 and 0, and high at values near 0.5.
    # So we decrease the weight of the error when the confidence is high, because we want to leave those values alone.

    l1_delta = l1_error * nonlin(l1,True)

    # update weights: It is all for this, to update our weights-matrix:
    # The updated weight is the value of the weight times the amount we want to change it.
    # this line does it for all 4 training examples at once:
    syn0 += np.dot(l0.T,l1_delta)

print "Output After Training:"
print l1


# The central idea is that weights from input 1 (or 0) to output 1 (or 0) are increased, and weights between wrong answers are penalized.
# The weights for perfectly correlated input will increment constantly; others will vary (as waves) as cancel each other out. Damn it really is all about circles.


# A harder problem is when there is *not* an input that is perfectly correlated to the output.
# This generates a *non-linear* system, because output depends on a relationship/combination of certain different inputs.
# This is like image-recognition: An individual pixel's value doesn't matter to whether it's a baby or a dog; but its relation to other pixels matters.

# In this case, we need two layers -- one to combine inputs, and one to map those to the outputs.
# If any of our cols have a correlation to the output, then that input gets better at mapping to that output, and our first layer gets better at mapping to that value in the second layer.




# This requires backpropagaion (stacking on top of each other two instances of the thing we just did):
def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in xrange(60000):

	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = y - l2
    
    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
