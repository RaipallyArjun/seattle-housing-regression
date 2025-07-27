#Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
df = pd.read_csv("../data/kc_house_data.csv")
df.head()
# Display data types of each column
df.describe()
# Counts values of floors
floor_counts = df["floors"].value_counts().to_frame()
floor_counts.columns = ['count']
print(floor_counts)
# Boxplot to check price outliers by waterfront view
sns.boxplot(x="waterfront", y="price", data=df)
plt.title("Price Distribution by Waterfront")
plt.show()
# regplot to check correlation between 'sqft_above' and 'price'
sns.regplot(x="sqft_above", y="price", data=df)
plt.title("Price vs Sqft Above")
plt.show()
# Linear Regression with single feature 'sqft_living'
lr = LinearRegression()
lr.fit(df[['sqft_living']], df['price'])
print("R^2 (sqft_living):", lr.score(df[['sqft_living']], df['price']))
# Linear Regression with multiple features
from sklearn.linear_model import LinearRegression
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement",
            "view", "bathrooms", "sqft_living15", "sqft_above",
            "grade", "sqft_living"]
X = df[features].dropna()
y = df.loc[X.index, 'price']
lr = LinearRegression()
lr.fit(X, y)
print("R^2 (multiple features):", lr.score(X, y))
# Train/Test split for model validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#  Pipeline - Scale, Polynomial Transform, Linear Regression
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
])
pipe.fit(X_train, y_train)
print("R^2 (Pipeline Polynomial Regression):", pipe.score(X_test, y_test))
# Ridge Regression (Linear)
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
print("R^2 (Ridge Regression):", ridge.score(X_test, y_test))
# Ridge Regression (Polynomial Features)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
ridge_poly = Ridge(alpha=0.1)
ridge_poly.fit(X_train_poly, y_train)
print("R^2 (Polynomial Ridge):", ridge_poly.score(X_test_poly, y_test))
