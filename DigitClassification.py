from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import os

# Load features
csv_file = "mnist_features_complete.csv"
if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found!")
    print("Please run the mnist_feature_extractor.py script first to generate the features.")
    exit(1)

print(f"Loading features from {csv_file}...")
df = pd.read_csv(csv_file)

# Step 1: Keep 20% of each class
balanced_subset = df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42))

# Step 2: Train-test split: 80% train, 20% test from the remaining
train_data, test_data = train_test_split(balanced_subset, test_size=0.2, stratify=balanced_subset['label'], random_state=42)

X_train = train_data.drop(columns=['label'])
y_train = train_data['label']
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

# Step 3: Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Step 4: Report results
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print(f"Dataset shape: {df.shape}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

print(f"\nPer-class accuracy:")
for digit in range(10):
    if str(digit) in report:
        precision = report[str(digit)]['precision']
        recall = report[str(digit)]['recall']
        f1 = report[str(digit)]['f1-score']
        print(f"Digit {digit}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

print(f"\nOverall metrics:")
print(f"Macro avg F1-score: {report['macro avg']['f1-score']:.4f}")
print(f"Weighted avg F1-score: {report['weighted avg']['f1-score']:.4f}")