import numpy as np
import pandas as pd

from itertools import chain, combinations

# Fijamos la semilla
rng = np.random.default_rng(seed=123)

# Construcción del df
mean = [0.1, -0.1]
cov = [[1,0],[0,1]]

x1, x2 = rng.multivariate_normal(mean, cov, 10).T
y = np.sign(x1+x2).astype(int)

data = pd.DataFrame({'x1': x1,'x2': x2,'target':y})

X = data[['x1','x2']]
y = data['target']

subsets = list(chain.from_iterable(combinations([1,2,3], r) for r in range(4)))
for subset in subsets:
    l = list(subset)
    print(f'Para el conjunto {l}')
    print(f'Obtnemos {X.iloc[l]}')
