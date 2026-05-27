import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Loading the data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

print(" Data Head ")
print(df.head())

# Splitting into training and testing sets
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Training and evaluating models
print("\n Model Results ")

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
print("KNN Accuracy:", knn_acc * 100, "%")

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy:", dt_acc * 100, "%")

# Logistic Regression
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", lr_acc * 100, "%")

# Printing confusion matrix for KNN
print("\n Confusion Matrix (KNN) ")
print(confusion_matrix(y_test, knn_pred))

# Visualizations

# Plot 1: Feature Scatter Plot
plt.figure()
sns.scatterplot(data=df, x="petal length (cm)", y="petal width (cm)", hue="species")
plt.title("Petal Length vs Width")
plt.savefig("petal_scatter.png")
plt.close()

# Plot 2: Accuracy Comparison Bar Chart
plt.figure()
models = ["KNN", "Decision Tree", "Logistic Regression"]
accuracies = [knn_acc * 100, dt_acc * 100, lr_acc * 100]
plt.bar(models, accuracies)
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")
plt.savefig("accuracy_comparison.png")
plt.close()

# Testing a single prediction
print("\n Test Prediction ")
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = knn.predict(sample)
print("Input:", sample)
print("Predicted Species ID:", prediction[0])
print("Predicted Species Name:", iris.target_names[prediction[0]])
