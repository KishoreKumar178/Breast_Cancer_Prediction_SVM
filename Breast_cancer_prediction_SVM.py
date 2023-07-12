#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,mean_squared_error,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold
#Loading dataset
data = pd.read_csv(r"D:\GUVI\casestudy\Project\cancer.csv")
#Encoding the categorical variable and removing dummy variable
data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
data = data.set_index('id')
del data['Unnamed: 32']
#selecting features and target
X = data.drop("diagnosis", axis = 1)
y = data["diagnosis"]
#Splitting Traing and Test data
x_train, x_test, y_train, y_test= train_test_split(X, y, test_size= 0.30, random_state=5)
# standardizing
st_x= StandardScaler()    
x_train_scaled= st_x.fit_transform(x_train)    
x_test_scaled= st_x.transform(x_test)  
#Fitting the model
classifier = SVC(kernel='linear', random_state=5)  
classifier.fit(x_train_scaled, y_train)  
# Predicting the model
y_test_pred_scaled= classifier.predict(x_test_scaled)  
y_train_pred_scaled= classifier.predict(x_train_scaled)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_test_pred_scaled)
precision = precision_score(y_test, y_test_pred_scaled, pos_label=1)
recall = recall_score(y_test, y_test_pred_scaled, pos_label=1)
f1 = f1_score(y_test, y_test_pred_scaled, pos_label=1)
auc_roc = roc_auc_score(y_test, y_test_pred_scaled)

# Print the performance metrics
print("*****************************Model Performance*****************************************")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"AUC-ROC: {auc_roc}")

#AUC ROC curve
train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred_scaled)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred_scaled)

plt.grid()

plt.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
plt.plot([0,1],[0,1],'g--')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC(ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()
# stratified K_fold cross validation
skf = StratifiedKFold(n_splits=5, shuffle= True, random_state= 17)
#hyperparameter tuning
classifier = SVC()
param_grid = {'C': [0.001,0.01,0.1,0.75,0.8,1,1.1,8,10,20,30,40],
              'gamma': [ 0.001, 0.01, 0.1, 1, 10, 100],
              'kernel': ['rbf','linear','poly']}
grid = GridSearchCV(classifier,param_grid, cv = skf)
grid.fit(x_train_scaled,y_train)
grid_test_svc = grid.predict(x_test_scaled)
grid_train_svc = grid.predict(x_train_scaled)
accuracy_score(y_test,grid_test_svc)
print(grid.best_params_)
print(grid.best_estimator_.get_params())
print(classification_report(y_test,grid_test_svc))
# Evaluate the model's performance
accuracy = accuracy_score(y_test, grid_test_svc)
precision = precision_score(y_test, grid_test_svc, pos_label=1)
recall = recall_score(y_test, grid_test_svc, pos_label=1)
f1 = f1_score(y_test, grid_test_svc, pos_label=1)
auc_roc = roc_auc_score(y_test, grid_test_svc)

# Print the performance metrics
print("*****************************Model Performance with hyperparameter tunning*****************************************")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"AUC-ROC: {auc_roc}")
#AUC ROC curve
train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, grid_train_svc)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, grid_test_svc)

plt.grid()

plt.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
plt.plot([0,1],[0,1],'g--')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC(ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()
# Bagging
model = BaggingClassifier(SVC(kernel = "rbf", random_state=5))
model.fit(x_train_scaled, y_train)
print(model.score(x_test_scaled,y_test))