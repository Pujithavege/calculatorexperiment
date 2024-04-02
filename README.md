!pip install decision-tree-id3
import matplotlib.pyplot as plt
import pandas as pd
import six
import sys
sys.modules['sklearn.externals.six']=six
from id3 import Id3Estimator

tennis_data=pd.read_csv('/content/drive/MyDrive/PlayTennis.csv')

tennis_data

tennis_data.head(5)

from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()
tennis_data['Outlook']=Le.fit_transform(tennis_data['Outlook'])
tennis_data['Temperature']=Le.fit_transform(tennis_data['Temperature'])
tennis_data['Humidity']=Le.fit_transform(tennis_data['Humidity'])
tennis_data['Wind']=Le.fit_transform(tennis_data['Wind'])
tennis_data['Play Tennis']=Le.fit_transform(tennis_data['Play Tennis'])

tennis_data

y=tennis_data['Play Tennis']
x=tennis_data.drop(['Play Tennis'],axis=1)

from sklearn.tree import DecisionTreeClassifier,export_text
clf=DecisionTreeClassifier(criterion='entropy')
clf.fit(x,y)

tree_rules=export_text(clf,feature_names=list(x.columns))
print(tree_rules)

from sklearn import tree
fig,ax=plt.subplots(figsize=(10,10))
tree.plot_tree(clf,fontsize=10)
plt.show()

from sklearn.tree import export_graphviz
import graphviz
dot_data=export_graphviz(clf,feature_names=list(x.columns),filled=True,rounded=True,special_characters=True)
graph=graphviz.Source(dot_data)
graph.render("decision_tree",format="png",cleanup=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from id3 import Id3Estimator,export_text
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)
estimator=Id3Estimator()
estimator.fit(X_train,y_train)
y_pred=estimator.predict(X_test)
print("Classification Report")
print(classification_report(y_test,y_pred))
print("Accuracy Score:",accuracy_score(y_test,y_pred))




tr=export_text(estimator.tree_,feature_names=list(X_train.columns))
print(tr)

------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Create a disease dataset with missing values
data = {'Age': [25, 30, None, 22, 30, 35, 32, None, 40, 35],
'BloodPressure': [120, 110, None, 130, 115, 125, None, 122, 118, 130],
'Cholesterol': ['High', 'Normal', 'High', 'Normal', None, 'High', 'Normal', 'Normal', 'High',
'Normal'],
'Disease': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No']}

df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('disease_dataset.csv', index=False)

# Read the disease dataset
df = pd.read_csv('disease_dataset.csv')

# Display dataset information
print("Dataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

DATE: ROLLNO:

PVPSIT, CSE MACHINE LEARNING LAB (20CS3652) Page no:
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nLast 5 rows of the dataset:")
print(df.tail())
#find duplicates
duplicates = df[df.duplicated()]
print("\nDuplicate Rows (excluding the first occurrence):")
print(duplicates)

# Find missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Separate input (X) and output (Y) variables using iloc
X = df.iloc[:, :-1] # All columns except the last one (Disease)
Y = df.iloc[:, -1] # Last column (Disease)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X[['Age', 'BloodPressure']] = imputer.fit_transform(X[['Age', 'BloodPressure']])

# Apply label encoding to 'Cholesterol' column
label_encoder = LabelEncoder()
X['Cholesterol'] = label_encoder.fit_transform(X['Cholesterol'])

# Split the dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

DATE: ROLLNO:

PVPSIT, CSE MACHINE LEARNING LAB (20CS3652) Page no:
# Apply feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Display the processed data
print("\nProcessed Data:")
print(X_train_scaled)
print(X_test_scaled)
print(Y_train)
print(Y_test)
--------------------------------------------------
# Read values for weights w1, w2, and bias b
w1 = float(input("Enter the value for w1: "))
w2 = float(input("Enter the value for w2: "))
b = float(input("Enter the value for b: "))
# Print the values
print(f"Values: w1 = {w1}, w2 = {w2}, b = {b}")
def activate(x):
return 1 if x >= 0 else 0
def train_perceptron(inputs, desired_outputs, learning_rate, epochs):
global w1, w2, b
for epoch in range(epochs):
total_error = 0
for i in range(len(inputs)):
A, B = inputs[i]
target_output = desired_outputs[i]
output = activate(w1 * A + w2 * B + b)
error = target_output - output
w1 += learning_rate * error * A
w2 += learning_rate * error * B
b += learning_rate * error
total_error += abs(error)
if total_error == 0:
break
print("Updated Weights:")

DATE: ROLLNO:

PVPSIT, CSE MACHINE LEARNING LAB (20CS3652) Page no:
print(f"w1: {w1}, w2: {w2}, b: {b}")
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
desired_outputs = [0, 0, 0, 1]
learning_rate = 0.1
epochs = 100
train_perceptron(inputs, desired_outputs, learning_rate, epochs)
for i in range(len(inputs)):
A, B = inputs[i]
output = activate(w1 * A + w2 * B + b)
print(f"Input: ({A}, {B}) Output: {output}")
