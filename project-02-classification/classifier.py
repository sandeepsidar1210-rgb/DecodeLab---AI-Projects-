#  DecodeLabs AI Internship - Project 2

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score


# ---- INPUT ----
iris = load_iris()
X = iris.data         # Features: sepal length, sepal width, petal length, petal width
Y = iris.target       # Labels: 0=Setosa, 1=Versicolor, 2=Virginica


print("-" * 55)
print("       DecodeLabs AI Project 2 — KNN Classifier")
print("-" * 55)

print("------Dataset Overview----------------------")
print(f"   Total Samples   : {X.shape[0]}") # X.shape gives the dimensions of the data table - x.shape[0] number of rows
print(f"   Features        : {X.shape[1]} (sepal_len, sepal_wid, petal_len, petal_wid)") #  x.shape[1] number of columns
print(f"   Classes         : {iris.target_names.tolist()}")
 #iris.target_names gives the actual class names — ['setosa', 'versicolor', 'virginica']. .tolist() converts it from a numpy array to a regular Python list for clean printing.
print(f"   Samples/Class   : 50 each (balanced dataset)")

# ----- PROCESS -----
# 1. Train - Test Split (80% train, 20% test)

X_train , X_test , Y_train , Y_test = train_test_split( X,Y, 
                                                    test_size=0.2 ,
                                                    random_state=42 ,
                                                    shuffle=True
                                                    )
#random_state=42 ensures same split every time you run --> split is always identical
#shuffle=True randomizes before splitting to remove order bias
# test_size=0.2 — 20% of the 150 samples (= 30 flowers) goes to testing. The remaining 80% (= 120 flowers) goes to training

print(f"\n── TRAIN-TEST SPLIT ─────────────────────────────────")
print(f"   Training Samples : {X_train.shape[0]} (80%)")
print(f"   Testing Samples  : {X_test.shape[0]} (20%)")


# 2. Feature Scaling (StandardScaler)

scaler = StandardScaler()
# Fit ONLY on training data, then transform both - (Fitting on test data would be "data leakage" — cheating)

X_train = scaler.fit_transform(X_train) # the scaler looks at all the training data and calculates the mean and standard deviation of each feature. 
# transform — it then uses those statistics to normalize every value. After this, each feature column has mean=0 and variance=1.

X_test  = scaler.transform(X_test)
# This is critical. We use the statistics learned from the training data to scale the test data. 

print(f"\n── FEATURE SCALING ─────────────────────────────────")
print(f"   Method : StandardScaler (Mean=0, Variance=1)")
print(f"   Status : Applied ✅")


#3.  Train the KNN Model (K=5, The Elbow Point)  Why K=5: K=1 overfits, K=100 underfits. K=5 is the sweet spot.

k = 5
model= KNeighborsClassifier(n_neighbors=k)

# FIT: Model memorizes the training data map
model.fit(X_train, Y_train)

print(f"\n── MODEL TRAINING ───────────────────────────────────")
print(f"   Algorithm : K-Nearest Neighbors")
print(f"   K Value   : {k} (majority vote from 5 nearest neighbors)")
print(f"   Status    : Trained ✅")


#-------Output - Predict and Validate----------
# predict : Apply learned logic to unseen test data

predictions = model.predict(X_test)
# We give it the 30 test flower measurements (X_test) that it has never seen before. It predicts a class (0, 1, or 2) for each one

# Accuracy (basic metric) ___________________________
acc = accuracy_score(Y_test, predictions)
# Compares our predictions against the real answers y_test


# F1 Score
f1 = f1_score(Y_test, predictions, average='weighted')
# average ='weighted' means it calculates F1 for each class separately and then averages them weighted by how many samples each class has.

# Confusion Matrix
cm = confusion_matrix(Y_test, predictions)

print(f"\n── OUTPUT VALIDATION ────────────────────────────────")
print(f"   Accuracy  : {acc * 100:.2f}%")
print(f"   F1 Score  : {f1:.4f}  (weighted average)")


print(f"\n── CONFUSION MATRIX ─────────────────────────────────")
print(f"   Rows = Actual Class | Columns = Predicted Class")
print(f"\n              Setosa  Versicolor  Virginica")


labels = ["Setosa    ", "Versicolor", "Virginica "]
for i, row in enumerate(cm):
    print(f"   {labels[i]}  {row}")

# enumerate(cm) loops through each row of the confusion matrix while also giving you the index i. We use i to get the class name from labels and row to print the actual numbers.    

print(f"\n── DETAILED CLASSIFICATION REPORT ───────────────────")

print(classification_report(Y_test, predictions, target_names = iris.target_names))

# Prints a full table showing precision, recall, and F1 score for each individual class. 

#Predicts a custom flower
print("── LIVE PREDICTION ──────────────────────────────────")
print("   Enter measurements of a flower to classify it.")
print("   (Press Enter to use a sample: [5.1, 3.5, 1.4, 0.2])\n")

raw = input("   sepal_length, sepal_width, petal_length, petal_width: ").strip()

if raw == "":
    sample = [5.1, 3.5, 1.4, 0.2] #setosa
else:
    sample = [ float(x) for x in raw.split(",")]    

# Scale the custom input using the SAME scaler fitted on training data

sample_scaled = scaler.transform([sample])
# We wrap sample in [] to make it a 2D array since transform expects that shape.
result = model.predict(sample_scaled)
prob   = model.predict_proba(sample_scaled)[0]


print(f"\n   Input     : {sample}")
print(f"   Predicted : {iris.target_names[result[0]].upper()} 🌸")
print(f"   Confidence: Setosa={prob[0]:.0%} | Versicolor={prob[1]:.0%} | Virginica={prob[2]:.0%}")
print("=" * 55)


# result[0] gives us the number (0, 1, or 2). We use it as an index into iris.target_names to get the actual flower name. .upper() makes it all caps for emphasis.

# prob[0], prob[1], prob[2] are the three probabilities. :.0% is a format specifier that converts a decimal like 0.6 into 60% with zero decimal places.
