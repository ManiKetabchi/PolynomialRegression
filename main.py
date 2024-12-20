import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

noise = 0.5
degreevar = 3

X = 4 * np.random.rand(100, 1) - 2
y = (X**degreevar) + (noise * np.random.randn(100, 1))

poly_features = PolynomialFeatures(degree=degreevar, include_bias=False)
X_poly = poly_features.fit_transform(X)

reg = LinearRegression()
reg.fit(X_poly, y)

X_vals = np.linspace(-2, 2, 100).reshape(-1, 1)
X_vals_poly = poly_features.transform(X_vals)


y_vals = reg.predict(X_vals_poly)

plt.scatter(X, y)
plt.plot(X_vals, y_vals, color='red')
plt.show()
