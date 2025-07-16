
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# PARAMETERS
CSV_PATH = "mnist_extended_features_no_scipy.csv"
LIMIT_PER_DIGIT = 5000  # Limit number of samples per digit

# Define weights for each feature
FEATURE_WEIGHTS = {
    'dark_pixel_count': 1.0,
    'avg_x': 1.0,
    'avg_y': 1.0,
    'bbox_height': 1.0,
    'bbox_width': 1.0,
    'intersection_count': 1.0,
    'loop_count': 1.0,
    'corner_count': 1.0,
    'symmetry_metric': 1.0,
    'writing_angle_global': 1.0,
    'writing_angle_tl': 1.0,
    'writing_angle_tr': 1.0,
    'writing_angle_bl': 1.0,
    'writing_angle_br': 1.0,
    'writing_magnitude_global': 1.0,
    'writing_magnitude_tl': 1.0,
    'writing_magnitude_tr': 1.0,
    'writing_magnitude_bl': 1.0,
    'writing_magnitude_br': 1.0
}

# Load CSV
df = pd.read_csv(CSV_PATH)

# Limit samples per digit
df_limited = df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min(len(x), LIMIT_PER_DIGIT), random_state=42)).reset_index(drop=True)

# Split features and labels
labels = df_limited['label']
features = df_limited.drop(columns=['label'])

# Normalize
scaler = MinMaxScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
features_scaled['label'] = labels.values

# Train/test split
train_df, test_df = train_test_split(features_scaled, test_size=0.2, stratify=features_scaled['label'], random_state=42)
X_train = train_df.drop(columns=['label'])
y_train = train_df['label']
X_test = test_df.drop(columns=['label'])
y_test = test_df['label']

# Class averages from training set
train_df['label'] = y_train
class_averages = train_df.groupby('label').mean()

# Classification
y_pred = []
for i in range(len(X_test)):
    sample = X_test.iloc[i].values
    distances = {
        digit: sqrt(np.sum((sample - avg_vector.values) ** 2))
        for digit, avg_vector in class_averages.iterrows()
    }
    predicted_label = min(distances, key=distances.get)
    y_pred.append(predicted_label)

# Accuracy and error calculation
correct = sum(int(p == t) for p, t in zip(y_pred, y_test))
total = len(y_test)
incorrect = total - correct
accuracy = correct / total
error_rate = incorrect / total

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Error Rate: {error_rate * 100:.2f}%")
print(f"Correct: {correct} / {total}")
print(f"Incorrect: {incorrect} / {total}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(df['label'].unique()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(df['label'].unique()))
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
