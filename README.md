# seattle-housing-regression
The Seattle Housing Regression &amp; Analysis Project is a data science project focused on understanding and predicting housing prices in King County, Seattle. It involves end-to-end steps from data cleaning and exploration to model building and evaluation. By using different regression techniques.
## 📁 Dataset

The dataset used is kc_house_data.csv, which includes information about house sales in the King County, Seattle area.

## 📊 Tasks Performed

Displayed data types and descriptive statistics

Dropped unnecessary columns: id and Unnamed: 0

Counted unique floor values using value_counts()

Created boxplots for price distribution by waterfront view

Created regression plots between sqft_above and price

Applied Linear Regression using sqft_living as feature

Applied Multiple Linear Regression with 11 key features

Built a pipeline for polynomial regression

Applied Ridge Regression (regularized)

Used Ridge + Polynomial Features to reduce overfitting

## ✅ R² Scores Observed

Model

R² Score

Simple Linear (sqft_living)

~0.49285

Multiple Linear (11 features)

~0.65769

Polynomial Pipeline (degree=2)

~0.75133

Ridge Regression (alpha=0.1)

~0.647

Ridge + Polynomial Features

~0.7

## 📈 Visualizations

Images and plots used in the analysis can be found in the images/ folder. These include:

Boxplot: price vs waterfront

Scatter plot with regression line: sqft_above vs price

## 🔹 Technologies Used

Python

Pandas

Matplotlib & Seaborn

Scikit-learn (LinearRegression, Ridge, Pipeline, PolynomialFeatures)
## 📁 Installation
pip install -r requirements.txt
## 🚀 Run the Notebook
jupyter notebook notebooks/seattle_housing_analysis.ipynb
