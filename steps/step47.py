if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero.models import MLP
from dezero import Variable, as_variable
import dezero.functions as F
import numpy as np

model = MLP((10, 3))

# print(model.l0)

# x = np.array([[0.2, -0.4]])
# y = model(x)
# print(y)

def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y

x = Variable(np.array([[0.2, -0.4]]))
y = model(x)
p = softmax1d(y)
print(y)
print(p)

