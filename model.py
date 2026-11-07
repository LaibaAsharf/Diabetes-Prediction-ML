import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv("dataset/diabetes.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())

df = df.drop("Pregnancies", axis=1)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
# feature&predictions
print(X.head())
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
# mean&std
print("Glucose Mean:", X_train["Glucose"].mean())
print("Glucose Std:", X_train["Glucose"].std())

# scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Logistic Regression Model
model = LogisticRegression()

model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")
# prediction for x test
y_pred = model.predict(X_test)
len(y_pred)
print(y_pred)
# comparison b/w real outcomes and predicted outcomes
for actual, predicted in zip(y_test, y_pred):
    result = "Correct" if actual == predicted else "Wrong"
    print("Actual:", actual, "Predicted:", predicted, result)

# check correct predicted values
correct = (y_test == y_pred).sum()

print("Correct predictions:", correct)
print("Total predictions:", len(y_test))
#check Accuracy
#Accuracy formula correct predicted values / total predicted value * 100 convert in percentage
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
# confusion matrix
# check actual value 0 predicted 0
# and 0 predicted 1, 1 predicted 0 ,1 predicted1
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

#Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))