#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so


# In[3]:


# Load the xlsx file
excel_data = pd.read_excel("Heart_Disease_Prediction.xlsx")
# Read the values of the file in the dataframe
df = pd.DataFrame(excel_data)


# In[4]:


df.info()


# In[5]:


df.describe().T


# In[11]:


from sklearn.preprocessing import LabelEncoder

# Assuming 'Presence' is a column with values like 'Present' or 'Absent'
le = LabelEncoder()
df['Heart Disease'] = le.fit_transform(df['Heart Disease'])
# Converts 'Present' to 1 and 'Absent' to 0

# Now you can calculate the correlation
corr = df.corr()


# In[12]:


corr = df.corr()
plt.figure(figsize=(df.shape[1],df.shape[1]))
sns.heatmap(corr,annot=True,cmap="coolwarm")


# In[14]:


from sklearn.feature_selection import VarianceThreshold
selctor = VarianceThreshold(threshold=0.1)
selctor.fit_transform(df)
selctor.get_feature_names_out()


# In[15]:


df_var = df[['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
       'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
       'ST depression', 'Slope of ST', 'Number of vessels fluro',
       'Thallium', 'Heart Disease']]


# # **SMOTE**
# 

# In[16]:


from collections import Counter
Counter(df['Heart Disease'])


# In[17]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
x_sm, y_sm = sm.fit_resample(df.drop(['Heart Disease'],axis=1), df['Heart Disease'])


# In[18]:


Counter(y_sm)


# In[19]:


df_sm = x_sm
df_sm['Heart Disease'] = y_sm


# In[21]:


corr_sm = df_sm.corr()


# In[22]:


tt = corr_sm['Heart Disease'].mean() * corr_sm['Heart Disease'].std()

corr_sm[ (corr_sm['Heart Disease'] >= tt) | (corr_sm['Heart Disease'] <= -1*tt) ]['Heart Disease'].index


# In[23]:


df_sm_corr = df_sm[['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'EKG results',
       'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST',
       'Number of vessels fluro', 'Thallium', 'Heart Disease']]


# # **Train**
# 

# In[13]:


from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

gnb = GaussianNB()
lg = LogisticRegression(max_iter = 2000)
tree = DecisionTreeClassifier(random_state=1)
knn = KNeighborsClassifier()
rf = RandomForestClassifier(random_state=1)
svc = SVC(probability = True)
xgb = XGBClassifier(random_state=1)
model = [gnb,lg,tree,knn,rf,svc,xgb]


# In[26]:


def mokhtar(df,model):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['Heart Disease'],axis=1), df['Heart Disease'], test_size=0.3, random_state=42)
    print("----------------------------------------------------------------------------------")
    print("first the cross validation")
    print("----------------------------------------------------------------------------------")

    for i in model:
        cv = cross_val_score(i,X_train,y_train,cv = 5)
        model_name = type(i).__name__
        print('name : ' + model_name)
        print(cv.mean())
        print('################')
    print("----------------------------------------------------------------------------------")
    print("second the accuracy for predict")
    print("----------------------------------------------------------------------------------")

    for i in model:
        i.fit(X_train,y_train)
        y_pred_t = i.predict(X_train)
        y_pred_s = i.predict(X_test)
        model_name = type(i).__name__
        print('name : ' + model_name)
        accuracy_t = accuracy_score(y_train, y_pred_t)
        print(f"Accuracy for train: {accuracy_t:.2f}")
        accuracy_s = accuracy_score(y_test, y_pred_s)
        print(f"Accuracy for test: {accuracy_s:.2f}")
        print("+++++++++++++++++==================+++++++++++++++++")


# In[27]:


mokhtar(df,model)


# In[29]:


mokhtar(df_sm,model)


# In[28]:


mokhtar(df_sm_corr,model)


# In[30]:


mokhtar(df_var,model)


# In[31]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Generate a synthetic dataset for binary classification
X, y = make_classification(
    n_samples=1000,    # Number of samples
    n_features=20,     # Number of features
    n_classes=2,       # Binary classification (2 classes)
    random_state=42    # For reproducibility
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Heart Disease'],axis=1), df['Heart Disease'], test_size=0.3, random_state=42)

# Scale the data for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),                                                   # Dropout for regularization
    Dense(32, activation='relu'),                                   # Hidden layer
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Make predictions
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

# Example output of first 5 predictions
print("Predicted Labels:", predictions[:5].flatten())
print("True Labels:", y_test[:5])

