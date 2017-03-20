#!/usr/bin/env python3
import matplotlib.pylab as plt
import numpy as np
from relu import relu
from sigmoid import sigmoid
from step import step_function


x = np.arange(-5.0, 5.0, 0.1)
plt.plot(x, step_function(x))
plt.plot(x, sigmoid(x))
plt.plot(x, relu(x))
plt.ylim(-0.1, 1.1)
plt.show()
