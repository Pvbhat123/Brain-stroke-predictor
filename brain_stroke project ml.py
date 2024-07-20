#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("full_data.csv")
data


# # exploratory data analysis

# In[133]:


data.info()


# In[4]:


data.isnull().sum()/len(data)*100


# In[5]:


data.shape


# # outlier analyzation

# In[26]:


sns.boxplot(x=data['bmi'])


# In[27]:


sns.boxplot(x=data['avg_glucose_level'])


# In[36]:


data['avg_glucose_level'].describe()


# In[37]:


data['bmi'].describe()


# In[ ]:


#we cannot remove those outliers as they are responsible for the stroke


# # encoding

# In[134]:


data['work_type'].unique()


# In[44]:


data['Residence_type'].unique()


# In[42]:


data['smoking_status'].unique()


# In[52]:


data['ever_married'].unique()


# In[41]:


data['gender'].unique()


# In[46]:


from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()


# In[51]:


gender=le.fit_transform(data['gender'])
work_type=le.fit_transform(data['work_type'])
Residence_type=le.fit_transform(data['Residence_type'])
ever_married=le.fit_transform(data['ever_married'])
smoking_status=le.fit_transform(data['smoking_status'])


# In[53]:


data['gender']=gender
data['smoking_status']=smoking_status
data['Residence_type']=Residence_type
data['ever_married']=ever_married
data['work_type']=work_type


# In[54]:


#preprocessed data
data


# In[55]:


data.info()


# # partioning->train and testing of data

# In[63]:


X=data.drop('stroke',axis=1)
Y=data['stroke']


# In[83]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=101)


# In[90]:


X_train


# In[91]:


X_test


# In[92]:


Y_train


# In[93]:


Y_test


# # normalizing the data

# In[94]:


data.describe()


# In[96]:


from sklearn.preprocessing import StandardScaler
std=StandardScaler()


# In[99]:


X_train_std=std.fit_transform(X_train)
X_test_std=std.transform(X_test)


# In[100]:


X_train_std


# In[101]:


X_test_std


# # training

# In[103]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[112]:


dt.fit(X_train_std,Y_train)


# In[107]:


dt.feature_importances_


# In[108]:


X_train.columns


# In[115]:


Y_pred=dt.predict(X_test_std)
Y_pred


# In[110]:


from sklearn.metrics import accuracy_score


# In[118]:


ac_dt=accuracy_score(Y_test,Y_pred)
ac_dt


# # logistic regression

# In[122]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[123]:


lr.fit(X_train_std,Y_train)


# In[130]:


Y_pred_lr=lr.predict(X_test_std)


# In[132]:


Y_pred_lr


# In[131]:


ac_lr=accuracy_score(Y_test,Y_pred_lr)
ac_lr


# # KNN

# In[135]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[136]:


knn.fit(X_train_std,Y_train)


# In[140]:


Y_pred_knn=knn.predict(X_test_std)


# In[141]:


ac_knn=accuracy_score(Y_test,Y_pred)
ac_knn


# # SVM

# In[142]:


from sklearn.svm import SVC
sv=SVC()
sv.fit(X_train_std,Y_train)


# In[143]:


Y_pred=sv.predict(X_test_std)


# In[144]:


ac_sv=accuracy_score(Y_test,Y_pred)
ac_sv


# In[145]:


ac_lr


# # Random Forest

# In[146]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[147]:


rf.fit(X_train_std,Y_train)


# In[ ]:


Y_pred=rf.predict(X_test_std)


# In[149]:


ac_rf=accuracy_score(Y_test,Y_pred)


# In[150]:


ac_rf


# In[151]:


ac_knn


# In[152]:


ac_dt


# In[153]:


ac_lr


# In[154]:


plt.bar(['Decision Tree','Logistic','KNN','Random Forest','SVM'],[ac_dt,ac_lr,ac_knn,ac_rf,ac_sv])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.show()


# In[2]:


import gradio as gr

def predict_brain_stroke(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
    # Here is an example logic for demonstration purposes, you should replace this with your own prediction logic
    if age > 60 and hypertension == 1 and heart_disease == 1 and smoking_status == 1:
        return "There might be a stroke. Please consult a doctor."
    else:
        return "No stroke is predicted."

iface = gr.Interface(
    fn=predict_brain_stroke,
    inputs=["number", "number", "number", "number", "number", "number", "number", "number", "number", "number"],
    outputs="text",
    title="Brain Stroke Predictor",
    description="Predicts the likelihood of a brain stroke based on certain factors."
)

iface.launch(share=True)

