import pandas as pd  # Import pandas for data manipulation and analysis
import numpy as np   # Import numpy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for creating visualizations
import seaborn as sns  # Import seaborn for enhanced visualizations

# Load the traffic dataset
df = pd.read_csv('./dataset/traffic.csv')
df  # Display the dataframe

# Display information about the dataframe (data types, memory usage)
df.info()
# Display summary statistics of the dataframe (mean, std, min, max, etc.)
df.describe()
# Check for missing values in the dataset
df.isna().sum()
# Check for duplicate records in the dataset
df.duplicated().sum()

# Convert DateTime column to pandas datetime format for feature extraction
df['DateTime'] = pd.to_datetime(df['DateTime'])
# Extract year from DateTime
df['Year'] = df['DateTime'].dt.year
# Extract month from DateTime
df['Month'] = df['DateTime'].dt.month
# Extract day from DateTime
df['Day'] = df['DateTime'].dt.day
# Extract hour from DateTime
df['Hour'] = df['DateTime'].dt.hour
# Extract minute from DateTime
df['Minute'] = df['DateTime'].dt.minute

# Remove original DateTime column as we've extracted its components
df.drop(columns=['DateTime'], inplace=True)
# Group data by hour and calculate average number of vehicles
hourly_avg_vehicles = df.groupby('Hour')['Vehicles'].mean().reset_index()

# Plot the distribution of vehicles by hour of day
plt.figure(figsize=(10, 6))
sns.barplot(x='Hour', y='Vehicles', data=hourly_avg_vehicles, color='skyblue')
plt.title('Estimated Vehicles Distribution by Time of Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Number of Vehicles')
plt.xticks(rotation=45)
plt.show()

# Create a box plot to visualize vehicle distribution by junction
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Junction', y='Vehicles')
plt.title('Box Plot of Vehicles by Junction')
plt.xlabel('Junction')
plt.ylabel('Vehicles')
plt.show()

# Plot histogram of vehicle distribution before outlier removal
plt.figure(figsize=(10, 6))
sns.histplot(df['Vehicles'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Vehicles (Before Outlier Removal)')
plt.xlabel('Number of Vehicles')
plt.ylabel('Frequency')
plt.show()

# Identify outliers using the Interquartile Range (IQR) method
Q1 = df['Vehicles'].quantile(0.25)  # First quartile (25th percentile)
Q3 = df['Vehicles'].quantile(0.75)  # Third quartile (75th percentile)
IQR = Q3 - Q1  # Interquartile range
lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers
# Find all outliers
outliers = df[(df['Vehicles'] < lower_bound) | (df['Vehicles'] > upper_bound)]
# Create a cleaned dataframe by removing outliers
cleaned_df = df[~df['Vehicles'].isin(outliers['Vehicles'])]

# Plot histogram of vehicle distribution after outlier removal
plt.figure(figsize=(10, 6))
sns.histplot(cleaned_df['Vehicles'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Vehicles (After Outlier Removal)')
plt.xlabel('Number of Vehicles')
plt.ylabel('Frequency')
plt.show()

# Define a function to categorize hours into time of day periods
def time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'

# Apply the time_of_day function to create a new categorical feature
df['TimeOfDay'] = df['Hour'].apply(time_of_day)
df.head()  # Display the first few rows of the dataframe
df  # Display the entire dataframe
df.columns  # Display all column names
df['Junction'].value_counts()  # Count occurrences of each junction
df['Hour']  # Display Hour column
df['TimeOfDay'].value_counts()  # Count occurrences of each time of day
# Remove unnecessary columns
df.drop(columns=['ID', 'Minute', 'Year'], inplace=True)
df.head()  # Display the first few rows after dropping columns

# Import LabelEncoder to convert categorical data to numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# Convert TimeOfDay categorical values to numeric (0: Afternoon, 1: Evening, 2: Morning, 3: Night)
df['TimeOfDay'] = le.fit_transform(df['TimeOfDay'])
df.head()  # Display the first few rows with encoded TimeOfDay

# Import Decision Tree and Random Forest for regression models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Separate features (X) and target variable (y)
X = df.drop(columns=['Vehicles'])  # Features
y = df['Vehicles']  # Target variable

# For Decision Tree feature importance
# Create and train a Decision Tree model
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X, y)
# Get feature importances
dt_importances = dt.feature_importances_
# Sort features by importance
dt_indices = np.argsort(dt_importances)[::-1]
dt_names = [X.columns[i] for i in dt_indices]

# Plot Decision Tree feature importance
plt.figure(figsize=(10, 6))
plt.title("Decision Tree Feature Importance")
plt.bar(range(X.shape[1]), dt_importances[dt_indices])
plt.xticks(range(X.shape[1]), dt_names, rotation=90)
plt.show()

# For Random Forest feature importance
# Create and train a Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
# Get feature importances
rf_importances = rf.feature_importances_
# Sort features by importance
rf_indices = np.argsort(rf_importances)[::-1]
rf_names = [X.columns[i] for i in rf_indices]

# Plot Random Forest feature importance
plt.figure(figsize=(10, 6))
plt.title("Random Forest Feature Importance")
plt.bar(range(X.shape[1]), rf_importances[rf_indices])
plt.xticks(range(X.shape[1]), rf_names, rotation=90)
plt.show()

# Remove TimeOfDay feature from the dataframe
df = df.drop('TimeOfDay',axis=1)
df  # Display dataframe after removing TimeOfDay

# Import libraries for model evaluation and cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Redefine features and target after removing TimeOfDay
X = df.drop(columns=['Vehicles'])
y = df['Vehicles']

# Create a dictionary of models to compare
# Using only 30 trees for Random Forest to improve training speed while maintaining good performance
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=30, random_state=42)
}

# Evaluate models using cross-validation with Mean Absolute Error metric
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    results[name] = np.mean(scores)  # Average MAE across all folds

# Print Mean Absolute Error for each model
for name, score in results.items():
    print(f"{name}: {abs(score)}")

# Create a bar chart comparing MAE for each model
model_names = list(results.keys())
mae_scores = [abs(score) for score in results.values()]

plt.figure(figsize=(14, 8))
bars = plt.bar(model_names, mae_scores, color='skyblue')

# Add text labels above each bar
for bar, score in zip(bars, mae_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{score:.2f}', ha='center', va='bottom')

plt.xlabel('Model')
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error for Different Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Evaluate models using R-squared (coefficient of determination)
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

# Import cross_val_predict for predicting values using cross-validation
from sklearn.model_selection import cross_val_predict

# Calculate Mean Absolute Percentage Error (MAPE) for each model
results_mape = {}
for name, model in models.items():
    y_pred = cross_val_predict(model, X, y, cv=5)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    results_mape[name] = mape

print("Mean Absolute Percentage Error (MAPE):")
for name, score in results_mape.items():
    print(f"{name}: {score:.2f}%")

# Create a bar chart comparing MAPE for each model
model_names = list(results_mape.keys())
mape_scores = list(results_mape.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, mape_scores, color='orange')

# Add text labels above each bar
for bar, score in zip(bars, mape_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{score:.2f}%', ha='center', va='bottom')

plt.xlabel('Model')
plt.ylabel('Mean Absolute Percentage Error (MAPE)')
plt.title('Mean Absolute Percentage Error (MAPE) for Different Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Determine the best model based on Mean Absolute Error
best_model_name = min(results, key=lambda k: abs(results[k]))
print(f"Best model based on MAE: {best_model_name}")

# Split data into training and testing sets for final model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X = df.drop(columns=['Vehicles'])
y = df['Vehicles']

# Split data: 70% for training, 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Select and initialize the best model based on MAE results
if best_model_name == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
elif best_model_name == "Decision Tree":
    model = DecisionTreeRegressor(random_state=42)
else:
    model = LinearRegression()

# Train the selected model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate MAE on the test set
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE) for {best_model_name}: {mae}")

# Plot actual vs predicted values for the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title(f'{best_model_name}: Actual vs Predicted Values')
plt.xlabel('Actual Number of Vehicles')
plt.ylabel('Predicted Number of Vehicles')
plt.grid(True)
# plt.show()  # Commented out to avoid redundant display

# Evaluate model using cross-validation
from sklearn.model_selection import cross_val_score

# Calculate MAE using 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
cv_scores = -cv_scores  # Convert negative MAE to positive
print("Cross-Validation Mean Absolute Error (MAE):", cv_scores.mean())
print("Cross-Validation Standard Deviation:", cv_scores.std())

# Re-calculate and print more detailed cross-validation statistics
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
cv_scores = -cv_scores
print("Cross-Validation MAE Scores:", cv_scores)
print("Min MAE:", cv_scores.min())
print("Max MAE:", cv_scores.max())

# Make predictions on the entire dataset
y_pred_all = model.predict(X)
# Calculate and print various performance metrics
print("Prediction Statistics:")
print(f"Mean Absolute Error: {mean_absolute_error(y, y_pred_all):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y, y_pred_all):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred_all)):.2f}")

# Calculate and print accuracy based on target range
mae = mean_absolute_error(y, y_pred_all)
y_range = y.max() - y.min()
accuracy_range = (1 - (mae / y_range)) * 100
print(f"Accuracy based on target range: {accuracy_range:.2f}%")

# Save the trained model to a file for later use
from joblib import dump
dump(model, 'traffic_model.pkl', protocol=4)

# Display the remaining columns in the dataframe
df.columns

# Interactive prediction loop to test the model
while True:
    # Get user input for prediction features
    junction = int(input("Enter junction number (1, 2, 3 or 4): "))
    month = int(input("Enter month (1-12): "))
    day = int(input("Enter day (1-31): "))
    hour = int(input("Enter hour (0-23): "))

    # Create a dataframe with user input
    input_data = pd.DataFrame({
        'Junction': [junction],
        'Month': [month],
        'Day': [day],
        'Hour': [hour]
    })

    # Make prediction using the trained model
    predicted_vehicles = model.predict(input_data)
    predicted_vehicles = round(predicted_vehicles[0])  # Round to nearest integer
    print("Predicted number of vehicles:", predicted_vehicles)

    # Ask user if they want to make another prediction
    cont = input("Do you want to predict again? (yes/no): ")
    if cont.lower() != 'yes':
        break  # Exit the loop if user doesn't want to continue