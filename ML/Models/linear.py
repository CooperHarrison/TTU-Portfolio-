# Linear regression example:
# - Loads x/y values from ML/Data collection/data2.csv, fits a straight-line
#   model, predicts y for x=4.5, and reports the test-set R-squared score.
# - Saves a scatter plot and fitted regression line to
#   linear_regression_plot.png, then displays the plot.
# - Run from the repository root with: py ML/Models/linear.py
# - Requires: py -m pip install pandas matplotlib scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df = pd.read_csv('ML/Data collection/data2.csv')

x = df[['x']].values
y = df['y'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)

sample_x = 4.5
prediction = model.predict([[sample_x]])[0]
print(f"Prediction for x={sample_x}: {prediction:.2f}")
print(f"R^2: {r2:.4f}")

x_order = sorted(df['x'].tolist())
line_y = model.predict([[value] for value in x_order])

plt.scatter(df['x'], df['y'], color='red', label='Actual data')
plt.plot(x_order, line_y, color='blue', linewidth=2, label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('linear_regression_plot.png', dpi=200)
print('Graph saved to linear_regression_plot.png')
plt.show()
