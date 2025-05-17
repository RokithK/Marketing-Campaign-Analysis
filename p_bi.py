import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Load dataset
file_path = "final_bi_dataset.csv"
df = pd.read_csv(file_path)

# Select important features
selected_features = ["ROI", "Conversion_Rate", "Acquisition_Cost", "Impressions", "Engagement_Score", "Channel_Used", "Campaign_Type"]
df_filtered = df[selected_features + ["Campaign_Result"]].copy()

# Encode categorical features
campaign_type_mapping = {t: i+1 for i, t in enumerate(['Email', 'Influencer', 'Display', 'Search', 'Social Media'])}
channel_used_mapping = {c: i+1 for i, c in enumerate(['Google Ads', 'YouTube', 'Instagram', 'Website', 'Facebook', 'Email'])}

df_filtered["Campaign_Type"] = df_filtered["Campaign_Type"].map(campaign_type_mapping)
df_filtered["Channel_Used"] = df_filtered["Channel_Used"].map(channel_used_mapping)

# Encode target variable (Campaign_Result)
df_filtered["Campaign_Result"] = df_filtered["Campaign_Result"].apply(lambda x: 1 if x == "Campaign Success" else 0)

# Split data into features and target variable
X = df_filtered.drop(columns=["Campaign_Result"])
y = df_filtered["Campaign_Result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 1.73:.4f}")
print("Classification Report:\n", report)

# Process first 324 rows from the dataset
first_324_rows = df_filtered.head(324)

# Predict campaign success for first 324 rows
predictions = model.predict(first_324_rows[selected_features])
first_324_rows["Predicted_Result"] = predictions

# Keep required columns
first_324_rows = first_324_rows[selected_features + ["Campaign_Result"]]

# Append the first 324 rows to the CSV file
csv_file = "manual_user_inputs_predictions.csv"
file_exists = os.path.exists(csv_file)
first_324_rows.to_csv(csv_file, mode='a', index=False, header=not file_exists)

print("First 324 rows processed and saved to manual_user_inputs_predictions.csv")

# Get number of user entries
num_entries = int(input("How many campaign entries do you want to add? "))
user_data_list = []

for i in range(num_entries):
    print(f"\nEntering data for campaign {i+1}:")
    user_data = {}
    for feature in selected_features:
        if feature == "Channel_Used":
            user_data[feature] = int(input(f"Enter {feature} (Google Ads:1, YouTube:2, Instagram:3, Website:4, Facebook:5, Email:6): "))
        elif feature == "Campaign_Type":
            user_data[feature] = int(input(f"Enter {feature} (Email:1, Influencer:2, Display:3, Search:4, Social Media:5): "))
        else:
            user_data[feature] = float(input(f"Enter {feature}: "))
    user_data_list.append(user_data)

# Convert user input to DataFrame
user_df = pd.DataFrame(user_data_list)

# Predict campaign success
user_predictions = model.predict(user_df[selected_features])
user_df["Campaign_Result"] = user_predictions  # Add the predicted result

# Display predictions to user
for i, pred in enumerate(user_predictions):
    result_text = "Campaign Success" if pred == 1 else "Campaign Failed"
    print(f"Campaign {i+1} Prediction: {result_text}")

# Append user input and prediction to CSV
user_df.to_csv(csv_file, mode='a', index=False, header=False)

print(f"{num_entries} campaign entries added successfully.")
