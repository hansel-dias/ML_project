#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# 
# # Heart failure Prediction Dataset

# ### Context
# Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.
# 
# People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help

# ### Attribute Information
# 
# 1. Age: age of the patient [years]
# 2. Sex: sex of the patient [M: Male, F: Female]
# 3. ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
# 4. RestingBP: resting blood pressure [mm Hg]
# 5. Cholesterol: serum cholesterol [mm/dl]
# 6. FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
# 7. RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
# 8. MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
# 9. ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
# 10. Oldpeak: oldpeak = ST [Numeric value measured in depression]
# 11. ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
# 12. HeartDisease: output class [1: heart disease, 0: Normal]</li>

# In[ ]:





#  ## 2 Data Analysis

# In[731]:


# libraries to be used for data analysis

import pandas as pd
import numpy as np

# Data Visualization libs

import matplotlib.pyplot as plt
import seaborn as sns

# for the yeo- jonson transormatio
import scipy.stats as stats


# In[732]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[733]:


heart_df = pd.read_csv('./heart.csv')


# In[734]:


heart_df.head()


# In[735]:


heart_df.shape


# ### 2.1 EDA

# In[736]:


heart_df.info()


# ### 2.2.1 lets get a feel of the data 

# In[737]:


heart_df.describe()


# ### 2.2.2. Check if there are any NULL Values present in the dataset
# 

# In[738]:


heart_df.isna().all()


# Thats great we dont have any null values in our dataset

# ### 2.2.3. Next lets see the type of data we have

# In[739]:


heart_df.isna().sum()


# In[740]:


heart_df.dtypes


# In[741]:


heart_df.head()


# In[742]:


heart_df.shape


# In[743]:


heart_df.info()


# In[744]:


heart_df.nunique()


# ### Dividing the data according to type for further feature engineering

# 1. **countinous**: Age, RestingBP, Cholestrol, MaxHR, Oldpeak
# 2. **Binary** : Sex, FastingBS, ExerciseAngina
# 3. **categorical** : ChestPaintype, RestingECG, ST_SLope

# In[784]:





# In[746]:


continuos_f = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
categorical_f = ["ChestPainType", "RestingECG", "ST_Slope"]
binaries_f = ["Sex", "FastingBS", "ExerciseAngina"]


# In[747]:


plt.style.use("seaborn")
plt.subplots_adjust(hspace=0.2)
color = 'winter'

fig, axs = plt.subplots(6, 2, figsize=(15,28))
i=1
for feature in heart_df.columns:
    if feature not in ["HeartDisease"] and i < 14:
        plt.subplot(6,2,i)
        sns.histplot(data=heart_df, x=feature, kde=True, palette=color, hue='HeartDisease')
        i+=1


# In[ ]:





# ### 2.1.1 Removing Outliers
# From the above distribution we can notice that columns **Cholestrol** and **RestingBP** contain some outliers, Lets Deal With them.
# 
#         For shall use the IQR method to handle the outliers
# 
# **IQR** - *IQR basiccally sets the upper and lower threshold for a feature, above or below which is considered as an outliers*
# 
# 
#    <img  src="https://editor.analyticsvidhya.com/uploads/12311IQR.png">
# 
# 
# 

#     capping the Outliers

# In[785]:


# Check for outliers
heart_df.plot(kind='box',subplots=True, sharex=False, 
        sharey=False,layout=(2, 7), figsize=(15,10));
       


# In[748]:


plt.figure(figsize=(20,5))
sns.boxplot(data=heart_df, x="Cholesterol")


# In[749]:


sns.histplot(data=heart_df, x='Cholesterol', kde=True, palette=color, hue='HeartDisease' )


# In[750]:


# function to detect outliers

def detect_outliers(col):
    q1 = heart_df[col].quantile(0.75)
    q2 = heart_df[col].quantile(0.25)
    print(f'75th percentile: {q1}')
    print(f'25th percentile: {q2}')
    
    IQR = q1 - q2

    upper_lim = q1 + 1.5 * IQR
    lower_lim = q2 - 1.5 * IQR


    print(f"Upper: {upper_lim}")
    print(f"Lower: {lower_lim}")

  # sreies of data containg bool values , indicating if data in Or out interquantile range
    interval = ((heart_df[col] > q2 - 1.5 * IQR) & (heart_df[col] < q1 + 1.5*IQR))

    print(f'Interval: {interval}')
    return heart_df[interval], heart_df[~interval] # df[true],df [false]

    # return heart_df[interval],df[~interval]

######################################

# not_out_df   # not outliers
# df_out  # ouliers dataframe
# heart_df.loc[69, 'Cholesterol']


def assign_mean(df_out, not_df_out, label=None):
    heart_df.loc[df_out[df_out["HeartDisease"] == 0].index, label] = not_df_out[not_df_out["HeartDisease"] == 0][label].mean()
    heart_df.loc[df_out[df_out["HeartDisease"] == 1].index, label] = not_df_out[not_df_out["HeartDisease"] == 1][label].mean()
    return

    
def delete_outliers(df_out):
    return heart_df.drop(df_out.index)


# In[751]:


not_out_df, df_out = detect_outliers('Cholesterol')
print(f'Outliers in cholesterol represent the {round((df_out.shape[0]*100)/heart_df.shape[0], 2)}% of our dataset')
# df_out_ch


# In[752]:


heart_df = delete_outliers(df_out[df_out["Cholesterol"] == 0])
assign_mean(df_out[df_out["Cholesterol"] != 0], not_out_df, 'Cholesterol')

plt.figure(figsize=(20,10))
sns.histplot(data=heart_df, x='Cholesterol', kde=True, palette=color, hue='HeartDisease')


# Resting Blood Pressure outliers

# In[753]:


plt.figure(figsize=(20,5))
sns.boxplot(data=heart_df,x='RestingBP')


# In[754]:


plt.figure(figsize=(20,10))
sns.histplot(data=heart_df, x='RestingBP', kde=True, palette=color, hue='HeartDisease')


# Histplot shows some outliers in **RestingBP** and it also needs some scaling too

# In[755]:


not_out_df_rb, df_out_rb = detect_outliers(col = 'RestingBP')
print(f'Outliers in RestingBP represent the {round((df_out_rb.shape[0]*100)/heart_df.shape[0], 2)}% of our dataset')


# In[756]:


heart_df = delete_outliers(df_out_rb)
plt.figure(figsize=(20,10))
sns.histplot(data=heart_df, x='RestingBP', kde=True, palette=color, hue='HeartDisease')


# ## Handling categorical Data

# In[757]:


# returns a list of catagorical col

cat_val = [cat for cat in heart_df.columns if heart_df[cat].dtype == 'object']


# In[758]:


cat_val


# In[759]:


def encode(cat_val):
    dataset = pd.DataFrame()
    for i in cat_val:
        df_cat = pd.get_dummies(heart_df[cat_val])
        dataset=pd.concat([dataset,df_cat],axis=1)
    return df_cat


# In[760]:


df_cat = encode(cat_val)


# In[761]:


final_df = heart_df.copy()


# In[762]:


final_df.drop(columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope' ],axis=1,inplace=True)


# In[763]:


final_df = pd.concat([final_df,df_cat],axis=1)


# In[764]:


# final_df = final_df.drop(['HeartDisease'],axis=1,inplace=True)


# In[765]:


final_df


# In[766]:


final_df.drop(['HeartDisease'],inplace=True,axis=1)


# In[767]:


X = final_df
y = heart_df['HeartDisease']


# In[768]:


final_df.describe()


# # Model Prep

# In[769]:


from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score,confusion_matrix,classification_report


# In[770]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
columns_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
X[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])

X.head()


# In[771]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[772]:


test_scores = []
train_scores = []

for i in range (1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)

    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))


# In[773]:


## score that comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))


# In[774]:


## score that comes from testing on the same datapoints that were used for training
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))


# In[775]:


knn.fit(X_train,y_train)
knn.score(X_test,y_test)


# In[776]:


params =[{'kernel':('linear', 'rbf'),'max_depth': list(range(10, 15)), 'max_features': list(range(0,14))}]


# In[777]:


prediction = knn.predict(X_test)
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))


# ### Hyperparameter Optimisation

# In[778]:


#In case of classifier like knn the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,y)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))


# In[780]:


import joblib


# In[781]:


joblib.dump(rfc, 'Random_forest_heart.joblib') 


# In[ ]:




