

#%% Libraries 

import pandas as pd 
import numpy as np 
import xgboost as xgb 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix 
import shap


# Importing the Keras libraries and packages
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense 
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

# Importing evaluation libraries
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix 
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, brier_score_loss



#%% EDA and Data Cleaning 

df = pd.read_csv(r'C:/Users/65904/Desktop/Machine Learning/Datasets/Telco_customer_churn.csv')
df.head()
df.drop(['Churn Label', 'Churn Score', 'CLTV', 'Churn Reason'],
        axis=1, inplace=True) ## set axis=0 to remove rows, axis=1 to remove columns
df.head()


# EDA
df['Count'].unique()
df['Country'].unique()
df['State'].unique()
df['City'].unique()
df.drop(['CustomerID', 'Count', 'Country', 'State', 'Lat Long'],
        axis=1, inplace=True) ## set axis=0 to remove rows, axis=1 to remove columns
df.head()


#  Replacing the white space in the city names with an underscore character `_`.

df['City'].replace(' ', '_', regex=True, inplace=True)
df.head()
df['City'].unique()[0:10]


# We also need to eliminate the whitespace in the column names, so we'll replace it with underscores. 
df.columns = df.columns.str.replace(' ', '_')
df.head()
df.dtypes
df['Phone_Service'].unique()
df['Total_Charges'].unique()


#  Dealing with blank spaces, `" "`, in the data.
len(df.loc[df['Total_Charges'] == ' '])
df.loc[df['Total_Charges'] == ' ']
df.loc[(df['Total_Charges'] == ' '), 'Total_Charges'] = 0
df.loc[df['Tenure_Months'] == 0]

df['Total_Charges'] = pd.to_numeric(df['Total_Charges'])
df.dtypes

df.replace(' ', '_', regex=True, inplace=True)
df.head()
df.size


#X,y Split
X = df.drop('Churn_Value', axis=1).copy() # alternatively: X = df_no_missing.iloc[:,:-1]
X.head()


y = df['Churn_Value'].copy()
y.head()


X.dtypes

# one hot encoding 
X_encoded = pd.get_dummies(X, columns=['City', 
                                       'Gender', 
                                       'Senior_Citizen', 
                                       'Partner',
                                       'Dependents',
                                       'Phone_Service',
                                       'Multiple_Lines',
                                       'Internet_Service',
                                       'Online_Security',
                                       'Online_Backup',
                                       'Device_Protection',
                                       'Tech_Support',
                                       'Streaming_TV',
                                       'Streaming_Movies',
                                       'Contract',
                                       'Paperless_Billing',
                                       'Payment_Method'])
X_encoded.head()

y.unique()

sum(y)/len(y)

# Training testing set split 
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)

sum(y_train)/len(y_train)
sum(y_test)/len(y_test)


#%% XGBoost
# # Build A Preliminary XGBoost Model

clf_xgb = xgb.XGBClassifier(objective='binary:logistic', 
                            eval_metric="logloss", 
                            seed=42, 
                            use_label_encoder=False)


## have ensured that the categorical values are all numeric, we do not expect XGBoost to do label encoding), so we set use_label_encoder=False


clf_xgb.fit(X_train, 
            y_train)


plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=["Did not leave", "Left"])


# param_grid = {
#     'max_depth': [3, 4, 5, 6, 7, 8],
#     'n_estimators': range(50, 250, 350),
#     'learning_rate': [0.1, 0.01, 0.05],
#     'gamma': [0, 0.25, 0.5, 1.0],
#     'reg_lambda': [0, 1.0, 10.0, 100.0], 
#     "min_child_weight" : [ 1, 3, 5, 7 ],
#     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
# }

# params={
#  "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
#  "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
#  "min_child_weight" : [ 1, 3, 5, 7 ],
#  "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
#  "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
# }


# Param 1
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.05],
    'gamma': [0, 0.25, 1.0],
    'reg_lambda': [0, 1.0, 10.0],
    'scale_pos_weight': [1, 3, 5] # NOTE: XGBoost recommends sum(negative instances) / sum(positive instances)
}
# Output: max_depth: 4, learning_rate: 0.1, gamma: 0.25, reg_lambda: 10, scale_pos_weight: 3

# Because learning_rate and reg_lambda were at the ends of their range, we will continue to explore those...

## Param 2
param_grid = {
    'max_depth': [4],
    'learning_rate': [0.1, 0.5, 1],
    'gamma': [0.25],
    'reg_lambda': [10.0, 20, 100],
      'scale_pos_weight': [3]
}
# Output: max_depth: 4, learning_rate: 0.1, reg_lambda: 10.

# NOTE: To speed up cross validiation, and to further prevent overfitting.
# We are only using a random subset of the data (90%) and are only
# using a random subset of the features (columns) (50%) per tree.


optimal_params = GridSearchCV(
                                estimator=xgb.XGBClassifier(objective='binary:logistic', 
                                eval_metric="logloss", ## this avoids a warning...
                                seed=42,
                                subsample=0.9,
                                colsample_bytree=0.5,
                                use_label_encoder=False),
                                param_grid=param_grid,
                                scoring='roc_auc', 
                                verbose=0, 
                                n_jobs = 10,
                                cv = 3
                                )

optimal_params.fit(X_train, 
                    y_train)
print(optimal_params.best_params_)


clf_xgb = xgb.XGBClassifier(seed=42,
                        objective='binary:logistic',
                        eval_metric="logloss", 
                        gamma=0.0,
                        learning_rate=0.05,
                        max_depth=4,
                        reg_lambda=1,
                        scale_pos_weight=3,
                        subsample=0.9,
                        colsample_bytree=0.5,
                        use_label_encoder=False)
clf_xgb.fit(X_train, 
            y_train)


print("\n-----Out of sample test: XGBoost")
predicted = clf_xgb.predict(X_test) 
Left = clf_xgb.predict_proba(X_test)
Left = [x[1] for x in Left] 

print("accuracy:", accuracy_score(y_test, predicted))
print("balanced_accuracy:", balanced_accuracy_score(y_test, predicted))
print("recall:", recall_score(y_test, predicted))
print("brier_score_loss:", brier_score_loss(y_test, Left))

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=["Did not leave", "Left"])


# We see that the optimized **XGBoost** model is a lot better at identifying people that left the company. 
# Of the **467** people that left the company, **373 (84%)**, were correctly identified. Before optimization, 
# we only correctly identified **245 (54%)**. However, this improvement was at the expense of not being able to correctly classify as many people that did not leave. 
# Before optimization, we correctly identified **1,166 (90%)** people that did not leave. Now we only correctly classify **946 (75%)**. 
# That said, this trade off may be better for the company because now it can focus resources on the people that leave if that will help them retain them.


# Now we get `shap` to create summaries of the data...
explainer = shap.Explainer(clf_xgb)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)



#%% Neural Network 
# Making an NN

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_dim=1178, units=589, kernel_initializer="uniform"))
# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=589, kernel_initializer="uniform"))
# Adding the third hidden layer
classifier.add(Dense(activation="relu", units=295, kernel_initializer="uniform"))
# Adding the fourth hidden layer
classifier.add(Dense(activation="relu", units=147, kernel_initializer="uniform"))
# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test,y_pred)

cm
accuracy