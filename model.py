import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./dataset/traffic.csv')
df

df.info()
df.describe()
df.isna().sum()
df.duplicated().sum()

df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Year'] = df['DateTime'].dt.year
df['Month'] = df['DateTime'].dt.month
df['Day'] = df['DateTime'].dt.day
df['Hour'] = df['DateTime'].dt.hour
df['Minute'] = df['DateTime'].dt.minute

df.drop(columns=['DateTime'], inplace=True)
hourly_avg_vehicles = df.groupby('Hour')['Vehicles'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Hour', y='Vehicles', data=hourly_avg_vehicles, color='skyblue')
plt.title('Estimated Vehicles Distribution by Time of Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Number of Vehicles')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Junction', y='Vehicles')
plt.title('Box Plot of Vehicles by Junction')
plt.xlabel('Junction')
plt.ylabel('Vehicles')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Vehicles'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Vehicles (Before Outlier Removal)')
plt.xlabel('Number of Vehicles')
plt.ylabel('Frequency')
plt.show()

Q1 = df['Vehicles'].quantile(0.25)
Q3 = df['Vehicles'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Vehicles'] < lower_bound) | (df['Vehicles'] > upper_bound)]
cleaned_df = df[~df['Vehicles'].isin(outliers['Vehicles'])]

plt.figure(figsize=(10, 6))
sns.histplot(cleaned_df['Vehicles'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Vehicles (After Outlier Removal)')
plt.xlabel('Number of Vehicles')
plt.ylabel('Frequency')
plt.show()

def time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'

df['TimeOfDay'] = df['Hour'].apply(time_of_day)
df.head()
df
df.columns
df['Junction'].value_counts()
df['Hour']
df['TimeOfDay'].value_counts()
df.drop(columns=['ID', 'Minute', 'Year'], inplace=True)
df.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['TimeOfDay'] = le.fit_transform(df['TimeOfDay'])
df.head()

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

X = df.drop(columns=['Vehicles'])
y = df['Vehicles']

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X, y)
importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]
names = [X.columns[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), names, rotation=90)
plt.show()

df = df.drop('TimeOfDay',axis=1)
df

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

X = df.drop(columns=['Vehicles'])
y = df['Vehicles']

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    results[name] = np.mean(scores)

for name, score in results.items():
    print(f"{name}: {abs(score)}")

model_names = list(results.keys())
mae_scores = [abs(score) for score in results.values()]

plt.figure(figsize=(14, 8))
bars = plt.bar(model_names, mae_scores, color='skyblue')

for bar, score in zip(bars, mae_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{score:.2f}', ha='center', va='bottom')

plt.xlabel('Model')
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error for Different Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

results_r2 = {}
for name, model in models.items():
    scores_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
    results_r2[name] = np.mean(np.abs(scores_r2))
print("R-squared Score:")
for name, score in results_r2.items():
    print(f"{name}: {score}")

print("R-squared Scores:")
for name, score in results_r2.items():
    print(f"{name}: {score:.2f}")

from sklearn.model_selection import cross_val_predict

results_mape = {}
for name, model in models.items():
    y_pred = cross_val_predict(model, X, y, cv=5)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    results_mape[name] = mape

print("Mean Absolute Percentage Error (MAPE):")
for name, score in results_mape.items():
    print(f"{name}: {score:.2f}%")

model_names = list(results_mape.keys())
mape_scores = list(results_mape.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, mape_scores, color='orange')

for bar, score in zip(bars, mape_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{score:.2f}%', ha='center', va='bottom')

plt.xlabel('Model')
plt.ylabel('Mean Absolute Percentage Error (MAPE)')
plt.title('Mean Absolute Percentage Error (MAPE) for Different Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X = df.drop(columns=['Vehicles'])
y = df['Vehicles']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Linear Regression: Actual vs Predicted Values')
plt.xlabel('Actual Number of Vehicles')
plt.ylabel('Predicted Number of Vehicles')
plt.grid(True)
plt.show()

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
cv_scores = -cv_scores
print("Cross-Validation Mean Absolute Error (MAE):", cv_scores.mean())
print("Cross-Validation Standard Deviation:", cv_scores.std())

cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
cv_scores = -cv_scores
print("Cross-Validation MAE Scores:", cv_scores)
print("Min MAE:", cv_scores.min())
print("Max MAE:", cv_scores.max())

y_pred_all = model.predict(X)
print("Prediction Statistics:")
print(f"Mean Absolute Error: {mean_absolute_error(y, y_pred_all):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y, y_pred_all):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred_all)):.2f}")

from joblib import dump
dump(model, 'traffic_model.pkl', protocol=4)

df.columns

while True:
    junction = int(input("Enter junction number (1, 2, 3 or 4): "))
    month = int(input("Enter month (1-12): "))
    day = int(input("Enter day (1-31): "))
    hour = int(input("Enter hour (0-23): "))

    input_data = pd.DataFrame({
        'Junction': [junction],
        'Month': [month],
        'Day': [day],
        'Hour': [hour]
    })

    predicted_vehicles = model.predict(input_data)
    predicted_vehicles = round(predicted_vehicles[0])
    print("Predicted number of vehicles:", predicted_vehicles)

    cont = input("Do you want to predict again? (yes/no): ")
    if cont.lower() != 'yes':
        break