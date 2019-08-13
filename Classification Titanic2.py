#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Classification Project
# In this project you will perform a basic classification task.
# You will apply what you learned about binary classification and tensorflow to implement a Kaggle project without much guidance. The challenge is to achieve a high accuracy score when trying to predict which passengers survived the Titanic crash. After building your model, you will upload your predictions to Kaggle and submit the score that you receive.


# In[ ]:


## Titanic: Machine Learning from Disaster

# [Kaggle](https://www.kaggle.com) has a [dataset](https://www.kaggle.com/c/titanic/data) containing the passenger list for the Titanic voyage. The data contains passenger features such as age, gender, and ticket class, as well as whether or not they survived.

# Your job is to load the data and create a binary classifier using TensorFlow to determine if a passenger survived or not. Then, upload your predictions to Kaggle and submit your accuracy score at the end of this colab, along with a brief conclusion.


# In[12]:


## Exercise 1: Create a Classifier
# 1. Download the [dataset](https://www.kaggle.com/c/titanic/data).
# 2. Load the data into this Colab.
# 3. Look at the description of the [dataset](https://www.kaggle.com/c/titanic/data) to understand the columns.
# 4. Explore the dataset. Ask yourself: are there any missing values? Do the data values make sense? Which features seem to be the most important? Are they highly correlated with each other?
# 5. Prep the data (deal with missing values, drop unnecessary columns, transform the data if needed, etc).
# 6. Split the data into testing and training set.
# 7. Create a `tensorflow.estimator.LinearClassifier`.
# 8. Train the classifier using an input function that feeds the classifier training data.
# 9. Make predictions on the test data using your classifier.
# 10. Find the accuracy, precision, and recall of your classifier.
get_ipython().system('pip install pyserial')


# In[16]:


get_ipython().system('pip install pandas')


# In[44]:


get_ipython().system('pip install matplotlib')


# In[46]:


get_ipython().system('pip install seaborn')


# In[47]:


# First load data into Colab
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

titanic_train = pd.read_csv('./train.csv')
titanic_train.head()


# In[48]:


# Look at description of dataset
titanic_train.describe()


# In[51]:


#Data Exploration


# In[49]:


# Let's see shape of data
print("dimension of titanic training data: {}".format(titanic_train.shape))


# In[50]:


# See if all data types make sense
titanic_train.dtypes


# In[52]:


# See how many NaN values per column
titanic_train.isnull().sum()


# In[53]:


# We can see a visual representation of the missing values
sns.heatmap(titanic_train.isnull(),yticklabels=False,cbar=False)
plt.show()


# In[ ]:


# Now let's look at the percentage of people that survived based on class


# In[55]:


titanic_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[56]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=titanic_train,palette='RdBu_r')
plt.show()


# In[57]:


# Look at distribution of age
sns.distplot(titanic_train['Age'].dropna(),kde=False,color='darkred',bins=30)
plt.show()


# In[58]:


# Look at distribution of fares on board
titanic_train['Fare'].hist(color='blue',bins=40,figsize=(8,4))
plt.show()


# In[ ]:


# We know many values for age are missing so well impute those based on mean for each class


# In[59]:


# box-and-whisker plots of age based on class
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=titanic_train,palette='winter')
plt.show()


# In[61]:


# Create function to impute age
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24
    else:
        return Age


# In[62]:


# Impute age
titanic_train['Age'] = titanic_train[['Age','Pclass']].apply(impute_age,axis=1)


# In[63]:


# Let's see if there are nay duplicate rows
titanic_train.duplicated().sum() > 0


# In[64]:


# Let's take a look at the correlation matrix
#Create Correlation df
corr = titanic_train.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(10, 10))
#Generate Color Map
colormap = sns.diverging_palette(220, 10, as_cmap=True)
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
#show plot
plt.show()


# In[ ]:


## Preparing the Data


# In[65]:


# Need to turn categorical variables into numerical values
from sklearn.preprocessing import LabelEncoder

# Encode sex
le = LabelEncoder()
le.fit(titanic_train.Sex)
titanic_train.Sex = le.transform(titanic_train.Sex)

# # Encode Embarked
# le = LabelEncoder()
# le.fit(titanic_train.Embarked)
# titanic_train.Embarked = le.transform(titanic_train.Embarked)

# titanic_train.head()
##########################################################################
dummy = pd.get_dummies(titanic_train['Embarked'])

# Create  a new_df which now contains dummy variables
titanic_train = pd.concat([titanic_train, dummy], axis=1)
titanic_train.drop(['Embarked'], axis=1, inplace=True)
titanic_train.head()


# In[66]:


# Correlation matrix again
#Create Correlation df
corr = titanic_train.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(10, 10))
#Generate Color Map
colormap = sns.diverging_palette(220, 10, as_cmap=True)
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
#show plot
plt.show()


# In[67]:


# Now according to the corr matrix, we'll only use
# sex, age, and fare as our features
final_df = titanic_train[['Sex', 'Pclass', 'Age', 'Survived']] # took out fare, put age back in
final_df.head()
final_df.columns[0]
#final_df.shape


# In[68]:


# Lets see value counts for how many people survived and how many died
final_df.Survived.value_counts()


# In[ ]:


## Split data for model


# In[69]:


from sklearn.model_selection import train_test_split

# X = final_df.iloc[:,:4]
# y = final_df.iloc[:, 4]
# #X.head()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df, test_df = train_test_split(
  final_df,
  stratify=final_df['Survived'],  
  test_size=0.2,
)


# In[ ]:


# Now we'll create our feature columns


# In[70]:


from tensorflow.feature_column import numeric_column
from tensorflow.feature_column import categorical_column_with_identity

feature_columns = []

for column_name in final_df.columns[:2]:
  #eature_columns.append(numeric_column(str(column_name)))
  feature_columns.append(categorical_column_with_identity(column_name, 3 if column_name=='Sex' else 2, default_value=1))
for column_name in final_df.columns[2:-1]:
  feature_columns.append(numeric_column(column_name))

feature_columns


# In[ ]:


# Find the number of classes


# In[71]:


# We know there are 2 classes
class_count = len(final_df['Survived'].unique())

class_count


# In[ ]:


## Create classifier


# In[77]:


get_ipython().system('pip install tensorflow')


# In[78]:


from tensorflow.estimator import LinearClassifier

classifier = LinearClassifier(feature_columns=feature_columns, n_classes=class_count)


# In[ ]:


## Train classifier


# In[79]:


import tensorflow as tf

from tensorflow.data import Dataset

# Instead of creating a df like in sklearn,
# in tf we have to create a function which returns a 
# tensorflow training dataset

features1 = ["Sex", "Pclass", "Age"] # took out age, fare
target = "Survived"

def training_input():
  features = {}
  # here set key and value is all values for that column in train_df 
  for i in train_df[features1]:
    features[i] = train_df[i]
 
  labels = train_df[target]

  training_ds = Dataset.from_tensor_slices((features, labels))
  training_ds = training_ds.shuffle(buffer_size=10000)
  training_ds = training_ds.batch(100)
  training_ds = training_ds.repeat(5) #Maybe increase????

  return training_ds

classifier.train(training_input)


# In[ ]:


## Make Test Predictions


# In[80]:


# create tf testing dataset
def testing_input():
  features = {}
  for i in train_df[features1]:
    features[i] = test_df[i]
  return Dataset.from_tensor_slices((features)).batch(1)

predictions_iterator = list(classifier.predict(testing_input))

predictions_iterator


# In[81]:


pred_probs = []
for p in predictions_iterator:
  pred_probs.append([p['probabilities'][0], p['probabilities'][1]])
print(pred_probs)


# In[ ]:


## Accuracy, Precision, and Recall


# In[82]:


for p in predictions_iterator:
  print(p.keys())
  print(p['logits'])
  print(p['probabilities'])
  print(p['class_ids'])
  print(p['classes'])
  break


# In[83]:


predictions_iterator = classifier.predict(testing_input)

predictions = [p['class_ids'][0] for p in predictions_iterator]


# In[84]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

prec = precision_score(test_df['Survived'], predictions)
recall = recall_score(test_df['Survived'], predictions)
accuracy = accuracy_score(test_df['Survived'], predictions)

print('Precision score: {}'.format(prec))
print('Recall score: {}'.format(recall))
print('Accuracy: {}'.format(accuracy))


# In[ ]:


## Exercise 2: Upload your predictions to Kaggle


# In[ ]:


# 1. Download the test.csv file from Kaggle and re-run your model using all of the training data.
# 2. Use this new test data to generate predictions using your model.
# 3. Follow the instructions in the [evaluation section](https://www.kaggle.com/c/titanic/overview/evaluation) to output the preditions in the format of the gender_submission.csv file. Download the predictions file from your Colab and upload it to Kaggle.


# **Written Response**

# Write down your conclusion along with the score that you got from Kaggle.


# In[85]:


# Download the new test.csv file
titanic_test = pd.read_csv('./test.csv')
titanic_test.head()


# In[ ]:


# Now re-run the model using all of the training data


# In[86]:


# Re-running model using all training data
import tensorflow as tf

from tensorflow.data import Dataset

# Instead of creating a df like in sklearn,
# in tf we have to create a function which returns a 
# tensorflow training dataset

features1 = ["Sex", "Pclass", "Age"] # took out fare, put age back in
target = "Survived"

def training_input():
  features = {}
  # here set key and value is all values for that column in train_df 
  for i in final_df[features1]:
    features[i] = final_df[i]
 
  labels = final_df[target]

  training_ds = Dataset.from_tensor_slices((features, labels))
  training_ds = training_ds.shuffle(buffer_size=10000)
  training_ds = training_ds.batch(100)
  training_ds = training_ds.repeat(5) #Maybe increase????

  return training_ds

classifier.train(training_input)


# In[ ]:


## Clean test data


# In[87]:


# See how many NaN values per column
titanic_test.isnull().sum()


# In[88]:


# box-and-whisker plots of age based on class
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=titanic_test,palette='winter')
plt.show()


# In[89]:


# Create function to impute age
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24
    else:
        return Age


# In[90]:


# Impute age
titanic_test['Age'] = titanic_test[['Age','Pclass']].apply(impute_age,axis=1)


# In[91]:


# Let's see if there are any duplicate rows
titanic_test.duplicated().sum() > 0


# In[92]:


# Need to turn categorical variables into numerical values
from sklearn.preprocessing import LabelEncoder

# Encode sex
le = LabelEncoder()
le.fit(titanic_test.Sex)
titanic_test.Sex = le.transform(titanic_test.Sex)

dummy = pd.get_dummies(titanic_test['Embarked'])

# Create  a new_df which now contains dummy variables
titanic_test = pd.concat([titanic_test, dummy], axis=1)
titanic_test.drop(['Embarked'], axis=1, inplace=True)
titanic_test.head()


# In[93]:


# Store the passenger ids in separate df
pass_id = titanic_test['PassengerId']
pass_id.head()


# In[94]:


# Just checking it stored all ids
for i in pass_id:
  print(i)


# In[95]:


# Now only keep the features that we need for model
final_test_df = titanic_test[['Sex', 'Pclass', 'Age']] # took out fare, put age back in
final_test_df.head()


# In[96]:


# Now that the test data is clean , we will use it to generate predictions using your model.


# In[97]:


# Use test.csv for testing dataset
def testing_input():
  features = {}
  for i in final_df[features1]:
    features[i] = final_test_df[i] ######################## changed test_def
  return Dataset.from_tensor_slices((features)).batch(1)

predictions_iterator = list(classifier.predict(testing_input))

predictions_iterator


# In[98]:


predictions_iterator = classifier.predict(testing_input)

predictions = [p['class_ids'][0] for p in predictions_iterator]


# In[99]:


# Make sure I have a prediction for every id 
print(len(predictions))
print(pass_id.shape)


# In[100]:


# Store ids in a list for later use
final_csv = []
pair = []
ids = []
for i in range(len(predictions)):
  ids.append(pass_id[i])


# In[101]:


# Use id and predictions arrays to make dictionary
final_csv_dict = {'PassengerId': ids, 'Survived': predictions}

# Turn dictionary into df
final_csv_df = pd.DataFrame.from_dict(final_csv_dict)
final_csv_df.head()


# In[102]:


# create csv to submit to kaggle
final_csv_df.to_csv('predictions.csv', encoding='utf-8', index=False)


# In[103]:


## Exercise 3: Improve your model


# In[104]:


# The predictions returned by the LinearClassifer contain scoring and/or confidence information about why the decision was made to classify a passenger as a survivor or not. Find the number used to make the decision and manually play around with different thresholds to build a precision vs. recall chart.


# In[105]:


y_test = []
for i in test_df['Survived']:
  y_test.append(i)
print(y_test)


# In[106]:


print(len(y_test))
print(len(pred_probs))


# In[107]:


y_score = [x[1] for x in pred_probs]


# In[108]:


from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve

average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score RF: {}'.format(average_precision))


# In[109]:


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))


# In[ ]:


Kaggle score was: 0.765555

