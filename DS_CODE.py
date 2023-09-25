'''
###########
## Code to impute data using sklearn
###########

import time
import psutil
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.contrib import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer


# Loading the data
data = pd.read_csv('/home/joyful/Downloads/higgs-boson/training.zip')
data.drop(['EventId', 'Weight'], axis=1, inplace=True)
data.replace(to_replace=-999, value=np.nan, inplace=True)
label_dict = {'b': 0, 's': 1}
data.replace({'Label': label_dict}, inplace=True)

# Check missing values in each column and calculate the percentage
missing_percentage = (data.isna().sum() / len(data)) * 100

# Sort columns by missing percentage in descending order
missing_percentage = missing_percentage.sort_values(ascending=False)

# List of columns with missing data
columns_with_missing_data = missing_percentage[missing_percentage > 0].index

# Create a copy of the data for imputation
data_imputed = data.copy()

# Loop through columns with missing data
for column in columns_with_missing_data:
    # Split data into known and unknown values
    known = data_imputed[data_imputed[column].notna()]
    unknown = data_imputed[data_imputed[column].isna()]

    # Features and target for the regression model
    X_known = known.drop(columns=[column])
    y_known = known[column]

    X_unknown = unknown.drop(columns=[column])

    # Initialize and train a machine learning model (HistGradientBoostingRegressor)
    regressor = HistGradientBoostingRegressor(random_state=42)
    regressor.fit(X_known, y_known)

    # Predict missing values
    imputed_values = regressor.predict(X_unknown)

    # Fill in missing values with imputed values
    data_imputed.loc[data_imputed[column].isna(), column] = imputed_values

# Save the imputed data to a CSV file
data_imputed.to_csv('/home/joyful/Downloads/4imputed_data.csv', index=True)
'''
###############################
## Logistic Reg.
##############################

import time
import psutil
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.contrib import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc


# Loading the data
data = pd.read_csv('/home/joyful/Downloads/higgs-boson/training.zip')
data.drop(['EventId', 'Weight'], axis=1, inplace=True)
data.replace(to_replace=-999, value=np.nan, inplace=True)
label_dict = {'b': 0, 's': 1}
data.replace({'Label': label_dict}, inplace=True)

# Train-test split
data_train, data_test = train_test_split(data, test_size=0.2, random_state=40)

# Columns with missing values with respective proportions
(data.isna().sum()[data.isna().sum() > 0] / len(data)).sort_values(ascending = False)

# Drop columns with more than 30% missing data
cols_missing_drop = [
    'DER_deltaeta_jet_jet',
    'DER_mass_jet_jet',
    'DER_prodeta_jet_jet',
    'DER_lep_eta_centrality',
    'PRI_jet_subleading_pt',
    'PRI_jet_subleading_eta',
    'PRI_jet_subleading_phi',
    'PRI_jet_leading_pt',
    'PRI_jet_leading_eta',
    'PRI_jet_leading_phi',
    'DER_mass_MMC'
]
#data_train.drop(cols_missing_drop, axis=1, inplace=True)
#data_test.drop(cols_missing_drop, axis=1, inplace=True)


# Median imputation for columns with missing values
for column in cols_missing_drop:
    median_value = data_train[column].median()
    data_train[column].fillna(median_value, inplace=True)
    data_test[column].fillna(median_value, inplace=True)


# Features-target split
X_train, y_train = data_train.drop('Label', axis=1), data_train['Label']
X_test, y_test = data_test.drop('Label', axis=1), data_test['Label']

# Min-Max normalization
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())


# Logistic function
def logistic(x):
    return 1 / (1 + np.exp(-x))


# Log loss
def log_loss(y, y_dash):
    return - (y * np.log(y_dash)) - ((1 - y) * np.log(1 - y_dash))

# Cost function
def cost_logreg_vec_reg(X, y, w, b, l):
    m = len(y)
    z = np.matmul(X, w) + (b * np.ones(m))
    y_dash = logistic(z)
    loss_vec = - (y * np.log(y_dash)) - ((1 - y) * np.log(1 - y_dash))
    cost = np.dot(loss_vec, np.ones(m)) / m
    cost += (l / (2 * m)) * np.dot(w, w)
    return cost

# Gradient descent algorithm
def grad_desc_reg(X, y, w, b, l, alpha, n_iter):
    m = len(y)
    cost_history = []

    for i in range(n_iter):
        z = np.matmul(X, w) + (b * np.ones(m))
        y_dash = logistic(z)

        grad_w = np.matmul(y_dash - y, X) / m + (l / m) * w
        grad_b = np.dot(y_dash - y, np.ones(m)) / m

        w -= alpha * grad_w
        b -= alpha * grad_b

        cost = cost_logreg_vec_reg(X, y, w, b, l)
        cost_history.append(cost)

        if i % (n_iter // 10) == 0 or i == n_iter - 1:
            print(f"Iteration {i:6}:    Cost  {cost:.4f}")

    return w, b, cost_history

# Initial values of the model parameters
w_init = np.zeros(X_train.shape[1])
b_init = 0.0

# Learning model parameters using gradient descent algorithm 
w_out, b_out, cost_history = grad_desc_reg(
    X_train.to_numpy(), y_train.to_numpy(), w_init, b_init, l=0.10, alpha=0.1, n_iter=500
)

# Plotting cost over iteration
plt.figure(figsize=(9, 6))
plt.plot(cost_history)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Cost", fontsize=14)
plt.title("Cost vs Iteration", fontsize=14)
plt.tight_layout()
plt.show()

# Prediction and evaluation 
y_test_prob = logistic(np.matmul(X_test.to_numpy(), w_out) + (b_out * np.ones(X_test.shape[0])))
y_test_pred = (y_test_prob > 0.5).astype(int)

y_train_prob = logistic(np.matmul(X_train.to_numpy(), w_out) + (b_out * np.ones(X_train.shape[0])))
y_train_pred = (y_train_prob > 0.5).astype(int)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Print final parameter values
print("Final Model Parameters:")
print("Coefficients (w_out):", w_out)
print("Intercept (b_out):", b_out)

# Training accuracy
train_accuracy = (y_train_pred == y_train.to_numpy()).mean()
print("Training Accuracy:", train_accuracy)

# Test accuracy
test_accuracy = (y_test_pred == y_test.to_numpy()).mean()
print("Test Accuracy:", test_accuracy)

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Perfect Learner')
plt.plot([0, 1], [1, 1], color='red', linestyle='--', label='No-Skill Learner')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.show()

# F1-Score for test data
f1_test = f1_score(y_test, y_test_pred)
print("F1-Score (Test):", f1_test)

x = np.linspace(-10, 10, 100)
y = logistic(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.xlabel('Input', fontsize=14)
plt.ylabel('Output', fontsize=14)
plt.title('Logistic Function', fontsize=14)
plt.grid(True)
plt.show()

y_true = 1  # Replace with the actual true label (0 or 1)
y_pred = np.linspace(0.01, 0.99, 100) 

loss = - (y_true * np.log(y_pred)) - ((1 - y_true) * np.log(1 - y_pred))

plt.figure(figsize=(8, 6))
plt.plot(y_pred, loss)
plt.xlabel('Predicted Probability', fontsize=14)
plt.ylabel('Log Loss', fontsize=14)
plt.title('Log Loss for a Single Example', fontsize=14)
plt.grid(True)
plt.show()


# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=.5, square=True, cbar=False)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Confusion Matrix', fontsize=14)
plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
plt.yticks([0, 1], ['True 0', 'True 1'])
plt.show()

