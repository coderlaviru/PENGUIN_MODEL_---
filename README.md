# PENGUIN_MODEL_---
This repository contains the code and data for the PENGUIN model, a machine learning project aimed at classifying different species of penguins based on their features. The model is built using Python and various machine learning libraries.

# Features:

- Data preprocessing and cleaning scripts
- Model training and evaluation scripts
- Jupyter notebooks for exploratory data analysis
- Pre-trained model files
- Detailed codes.

# 1. Know the data
# Importing libraries


```python
import numpy as np
import pandas as pd

#Importing tools for visualization 
import matplotlib.pyplot as plt 
import seaborn as sns 
#Import evaluation metric librarie s
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report 
from sklearn.preprocessing import LabelEncoder
#Libraries used for data  prprocessing 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

#Library used for ML Model implementation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb #Xtreme Gradient Boosting
#librries used for ignore warnings 
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```

# Dataset Loading


```python
penguin=pd.read_csv(r"C:\Users\lakshita\Downloads\archive\penguins.csv")
# Load the penguin dataset
penguin = sns.load_dataset("penguins")

# Drop rows with missing values
penguin = penguin.dropna()

# Select only the numerical columns for correlation matrix
penguin_numeric = penguin.select_dtypes(include=['float64', 'int64'])
```

# Dataset First View


```python
penguin.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.3</td>
      <td>20.6</td>
      <td>190.0</td>
      <td>3650.0</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>




```python
penguin.tail()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>338</th>
      <td>Gentoo</td>
      <td>Biscoe</td>
      <td>47.2</td>
      <td>13.7</td>
      <td>214.0</td>
      <td>4925.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>340</th>
      <td>Gentoo</td>
      <td>Biscoe</td>
      <td>46.8</td>
      <td>14.3</td>
      <td>215.0</td>
      <td>4850.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>341</th>
      <td>Gentoo</td>
      <td>Biscoe</td>
      <td>50.4</td>
      <td>15.7</td>
      <td>222.0</td>
      <td>5750.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>342</th>
      <td>Gentoo</td>
      <td>Biscoe</td>
      <td>45.2</td>
      <td>14.8</td>
      <td>212.0</td>
      <td>5200.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>343</th>
      <td>Gentoo</td>
      <td>Biscoe</td>
      <td>49.9</td>
      <td>16.1</td>
      <td>213.0</td>
      <td>5400.0</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>




```python
penguin.describe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>333.000000</td>
      <td>333.000000</td>
      <td>333.000000</td>
      <td>333.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>43.992793</td>
      <td>17.164865</td>
      <td>200.966967</td>
      <td>4207.057057</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.468668</td>
      <td>1.969235</td>
      <td>14.015765</td>
      <td>805.215802</td>
    </tr>
    <tr>
      <th>min</th>
      <td>32.100000</td>
      <td>13.100000</td>
      <td>172.000000</td>
      <td>2700.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>39.500000</td>
      <td>15.600000</td>
      <td>190.000000</td>
      <td>3550.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>44.500000</td>
      <td>17.300000</td>
      <td>197.000000</td>
      <td>4050.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.600000</td>
      <td>18.700000</td>
      <td>213.000000</td>
      <td>4775.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>59.600000</td>
      <td>21.500000</td>
      <td>231.000000</td>
      <td>6300.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Dataset rows and column count


```python
# Correcting the use of iris.info()
print(penguin.info())
print("Number of rows ", penguin.shape[0])
print("Number of Columns ",penguin.shape[1])
print(penguin.head(150))
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 333 entries, 0 to 343
    Data columns (total 7 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   species            333 non-null    object 
     1   island             333 non-null    object 
     2   bill_length_mm     333 non-null    float64
     3   bill_depth_mm      333 non-null    float64
     4   flipper_length_mm  333 non-null    float64
     5   body_mass_g        333 non-null    float64
     6   sex                333 non-null    object 
    dtypes: float64(4), object(3)
    memory usage: 20.8+ KB
    None
    Number of rows  333
    Number of Columns  7
           species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  \
    0       Adelie  Torgersen            39.1           18.7              181.0   
    1       Adelie  Torgersen            39.5           17.4              186.0   
    2       Adelie  Torgersen            40.3           18.0              195.0   
    4       Adelie  Torgersen            36.7           19.3              193.0   
    5       Adelie  Torgersen            39.3           20.6              190.0   
    ..         ...        ...             ...            ...                ...   
    151     Adelie      Dream            41.5           18.5              201.0   
    152  Chinstrap      Dream            46.5           17.9              192.0   
    153  Chinstrap      Dream            50.0           19.5              196.0   
    154  Chinstrap      Dream            51.3           19.2              193.0   
    155  Chinstrap      Dream            45.4           18.7              188.0   
    
         body_mass_g     sex  
    0         3750.0    Male  
    1         3800.0  Female  
    2         3250.0  Female  
    4         3450.0  Female  
    5         3650.0    Male  
    ..           ...     ...  
    151       4000.0    Male  
    152       3500.0  Female  
    153       3900.0    Male  
    154       3650.0    Male  
    155       3525.0  Female  
    
    [150 rows x 7 columns]
    

# Dataset information 


```python
#Checking information about the dataset using information 
penguin.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 333 entries, 0 to 343
    Data columns (total 7 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   species            333 non-null    object 
     1   island             333 non-null    object 
     2   bill_length_mm     333 non-null    float64
     3   bill_depth_mm      333 non-null    float64
     4   flipper_length_mm  333 non-null    float64
     5   body_mass_g        333 non-null    float64
     6   sex                333 non-null    object 
    dtypes: float64(4), object(3)
    memory usage: 20.8+ KB
    

# Duplicate rows


```python
dup=penguin.duplicated().sum()
print(f'Number of duplicate rows:{dup}')
```

    Number of duplicate rows:0
    

# Dropping duplicate rows


```python
#dropping duplicate rows
penguin=penguin.drop_duplicates()
```

# After Droppng Duplicates


```python
#Checking the number of rows again to see if duplicates were dropped
print(f'Number of rows after dropping duplicates : {penguin.shape[0]}')
```

    Number of rows after dropping duplicates : 333
    

# Count Plot For The Species Distribution


```python
#count plot for the species distribution
sns.countplot(data=penguin,x='bill_length_mm')
plt.title('Distribution of Penguin Species')
plt.show()
```


    
![png](images/output_19_0.png)
    



```python
#Scatter plot for bill_length_mm vs bill_depth_mm
sns.scatterplot(data=penguin,x='bill_length_mm',y='bill_depth_mm',hue='species')
plt.title('bill_length_mm vs bill_depth_mm')
plt.show()
```


    
![png](images/output_20_0.png)
    



```python
# Pair plot for visualizing relationships between all features
sns.pairplot(penguin, hue='species')
plt.show()
```


    
![png](images/output_21_0.png)
    



```python
#checking number of rows and column of the dataset using shape
print("Number of rows:",penguin.shape[0])
print("Number of columns:",penguin.shape[1])
```

    Number of rows: 333
    Number of columns: 7
    

# Missing Value/NULL Values


```python
penguin.isnull().sum()
```




    species              0
    island               0
    bill_length_mm       0
    bill_depth_mm        0
    flipper_length_mm    0
    body_mass_g          0
    sex                  0
    dtype: int64



# 2. Understanding The Variables


```python
#Dataset Columns
penguin.columns
```




    Index(['species', 'island', 'bill_length_mm', 'bill_depth_mm',
           'flipper_length_mm', 'body_mass_g', 'sex'],
          dtype='object')




```python
penguin.describe(include='all').round(2)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>333</td>
      <td>333</td>
      <td>333.00</td>
      <td>333.00</td>
      <td>333.00</td>
      <td>333.00</td>
      <td>333</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>3</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Adelie</td>
      <td>Biscoe</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>146</td>
      <td>163</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>168</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>43.99</td>
      <td>17.16</td>
      <td>200.97</td>
      <td>4207.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.47</td>
      <td>1.97</td>
      <td>14.02</td>
      <td>805.22</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>32.10</td>
      <td>13.10</td>
      <td>172.00</td>
      <td>2700.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>39.50</td>
      <td>15.60</td>
      <td>190.00</td>
      <td>3550.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.50</td>
      <td>17.30</td>
      <td>197.00</td>
      <td>4050.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>48.60</td>
      <td>18.70</td>
      <td>213.00</td>
      <td>4775.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>59.60</td>
      <td>21.50</td>
      <td>231.00</td>
      <td>6300.00</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



# Check Unique Values For EachVariables


```python
# Check unique values for each variables
for i in penguin.columns.tolist():
    print(" Number of unique values in",i,"is",penguin[i].nunique())
```

     Number of unique values in species is 3
     Number of unique values in island is 3
     Number of unique values in bill_length_mm is 163
     Number of unique values in bill_depth_mm is 79
     Number of unique values in flipper_length_mm is 54
     Number of unique values in body_mass_g is 93
     Number of unique values in sex is 2
    

# 3. Data Wrangling
# Data Wrangling Code


```python
# IF We don't need the 1st column so lets drop that 
penguin = penguin.drop('island', axis=1)  # Drop the 'island' column
```


```python
# New updated dataset
penguin.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Adelie</td>
      <td>39.3</td>
      <td>20.6</td>
      <td>190.0</td>
      <td>3650.0</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>



# 4. Data Vizualization,Storytelling & Experimenting With Charts:Understand The Relationships Between Variables
# Chart 1 : Distribution Of Numerical Variables


```python
# chart 1 histrogram visual code for distribution of numerical variables
#create a figure with subject 
plt.figure(figsize=(8,6))
plt.suptitle('Distribution of Penguin Bill Measurement',fontsize=14)

#create a 2x2 grid of subplot
plt.subplot(2,2,1) # subject 1 (top-left)
plt.hist(penguin['bill_length_mm'])
plt.title('Bill Length Distribution')

plt.subplot(2,2,2) #subject 2 (top-right)
plt.hist(penguin['bill_depth_mm'])
plt.title('Bill Depth Distribution')

plt.subplot(2,2,3) #subject 3 (bottom-left)
plt.hist(penguin['flipper_length_mm'])
plt.title('Flipper Length Distribution')

plt.subplot(2,2,4) #subject 4 (bottom-right)
plt.hist(penguin['body_mass_g'])
plt.title('Body Mass Distribution')

#display the subjects
plt.tight_layout()
plt.show()
```


    
![png](images/output_34_0.png)
    


# Chart - 2 Scatter plot visualization code for Bill Length vs Bill depth.


```python
#Define colours for each species and te corresponding species labels.
colors=['Purple','blue','pink']
species=['Adelie','Gentoo','Chinstrap']
```


```python
# Create a scatter plot for bill_length_mm vs bill_depth_mm for each species
for i in range(len(species)):
    #select data for the current species
    x=penguin[penguin['species']==species[i]]
    #create a scatter plot with the specified colors and labels for the current species.
    plt.scatter(x['bill_length_mm'],x['bill_depth_mm'],c=colors[i],label=species[i])

#Add labels to the x and y axes
plt.xlabel('Bill Lenght')
plt.ylabel('Bill Depth')

#Add a legend to identify species based on colors
plt.legend()

#Display the scatter plot
plt.show()
```


    
![png](images/output_37_0.png)
    


# Chart - 3 Scatter plot visualization code for Flipper Length vs Body Mass .


```python
#Define colours for each species and te corresponding species labels.
colors=['black','grey','yellow']
species=['Adelie','Gentoo','Chinstrap']
```


```python
# create a scatter plot for flipper_length_mm vs body_mass_g for each species
for i in range(len(species)):
    # Select data for the current species.
    x = penguin[penguin['species'] == species[i]]

    # Create a scatter plot with the specified color and label for the current species.
    plt.scatter(x['flipper_length_mm'], x['body_mass_g'], c=colors[i], label=species[i])

# Add labels to the x and y axes
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Body Mass (g)')

# Add a legend to identify species based on colors.
plt.legend()

# Display the scatter plot.
plt.show()
```


    
![png](images/output_40_0.png)
    


# Chart - 4 Scatter plot visualization code for Bill Length vs Bill depth.


```python
color=['yellow','pink','blue']
species=['Adelie','Gentoo','Chinstrap']
```


```python
for i in range(len(species)):
    #select data for the current species
    x=penguin[penguin['species']==species[i]]
    #create a scatter plot with the specified colors and labels for the current species.
    plt.scatter(x['bill_length_mm'],x['flipper_length_mm'],c=color[i],label=species[i])

#Add labels to the x and y axes
plt.xlabel('Bill Lenght')
plt.ylabel('Flipper Length')

#Add a legend to identify species based on colors
plt.legend()

#Display the scatter plot
plt.show()
```


    
![png](images/output_43_0.png)
    


# Chart 5 : Bill depth vs body mass


```python
color=['black','grey','pink']
species=['Adelie','Gentoo','Chinstrap']
```


```python
# create a scatter plot for flipper_length_mm vs body_mass_g for each species
for i in range(len(species)):
    # Select data for the current species.
    x = penguin[penguin['species'] == species[i]]

    # Create a scatter plot with the specified color and label for the current species.
    plt.scatter(x['bill_depth_mm'], x['body_mass_g'], c=color[i], label=species[i])

# Add labels to the x and y axes
plt.xlabel('Bill Depth')
plt.ylabel('Body Mass')

# Add a legend to identify species based on colors.
plt.legend()

# Display the scatter plot.
plt.show()
```


    
![png](images/output_46_0.png)
    


# Chart 6 Correlation HeatMap Visualization Code


```python
# Correlation Heatmap Visualization Code
corr_matrix = penguin_numeric.corr()

# Plot Heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(corr_matrix, annot=True, cmap='Reds_r')

# Setting Labels
plt.title('Correlation Matrix Heatmap')

# Display Chart
plt.show()
```


    
![png](images/output_48_0.png)
    


# 5. Feature Engineering & Data Preprocessing 
# Categorical Encoding



```python
#Encode the categorical columns
#create a LabelEncoder object
le=LabelEncoder()

#Encode the "species"column to convert the species names to numerical names to numerical labels
penguin['species']=le.fit_transform(penguin['species'])

#check the unique values in the 'species' column after encoding
unique_species=penguin['species'].unique()

#Display the unique encoded values
print("Encoded Species Value:")
print(unique_species)   #'Adelie'==0,'Gentoo'==1,'Chinstrap'==2

# One-hot encode the 'sex' column
penguin = pd.get_dummies(penguin, columns=['sex'], drop_first=True)
# The 'sex' column will be replaced with a single binary column, 'sex_Male'
```

    Encoded Species Value:
    [0 1 2]
    

# Data Scaling


```python
# Defining the X and y
x=penguin.drop(columns=['species'], axis=1)
y=penguin['species']
```

# 3. Data Splitting


```python
# Splitting the data to train and test
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3)
```


```python
# Checking the train distribution of dependent variable
y_train.value_counts()
```




    species
    0    111
    2     73
    1     49
    Name: count, dtype: int64



# 6. ML Model Implementation


```python
def evaluate_model(model,x_train,x_test,y_train,y_test):
    '''The function will take model, x train , x test , y train,y test
    and then it will fit the model ,then make predictions on the trained model,
    it will then print roc-auc score of train and test ,then plot the roc , auc curve,
    print the confusion matrix for train and test ,then print classification report for trin and test,
    then plot the feature importances if the model has feature importances,
    and finally it will return the following scores as a list:
    recall_train,recall_test,acc_train,acc_test ,F1_train,F1_test
    '''

    #fit the models to training data.
    model.fit(x_train,y_train)

    # make predictions on the test data 
    y_pred_train=model.predict(x_train)
    y_pred_test = model.predict(x_test)

    #calculate confusion matrix
    cm_train = confusion_matrix(y_train,y_pred_train)
    cm_test = confusion_matrix(y_pred_test,y_pred_test)

    fig,ax=plt.subplots(1,2,figsize=(11,4))

    print("\nConfusion Matrix")
    sns.heatmap(cm_train,annot=True,xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'],cmap="Greens",fmt='.4g',ax=ax[0])
    ax[0].set_xlabel("Peridicted Label")
    ax[0].set_ylabel("True Label")
    ax[0].set_title("Train Confusion Matrix ")

    sns.heatmap(cm_train,annot=True,xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'],cmap="Greens",fmt='.4g',ax=ax[1])
    ax[1].set_xlabel("Peridicted Label")
    ax[1].set_ylabel("True Label")
    ax[1].set_title("Test Confusion Matrix ")

    plt.tight_layout()
    plt.show()

    # calculate classification report
    cr_train=classification_report(y_train,y_pred_train,output_dict=True)
    cr_test=classification_report(y_test,y_pred_test,output_dict=True)
    print("\nTrain Classification Report:")
    crt2 = pd.DataFrame(cr_test).T
    print(crt2.to_markdown())

    # sns.heatmap(pd.DataFrame(cr_train).T.iloc[:, :-1], annot=True, cmap="Blues")
    print("\nTest Classification Report:")
    crt2 = pd.DataFrame(cr_test).T
    print(crt2.to_markdown())

    # sns.heatmap(pd.DataFrame(cr_test).T.iloc[:, :-1], annot=True, cmap="Blues")
    precision_train = cr_train['weighted avg']['precision']
    precision_test = cr_test['weighted avg']['precision']
    
    recall_train = cr_train['weighted avg']['recall']
    recall_test = cr_test['weighted avg']['recall']
    
    acc_train = accuracy_score(y_true = y_train, y_pred = y_pred_train)
    acc_test = accuracy_score(y_true = y_test, y_pred = y_pred_test)

    F1_train = cr_train['weighted avg']['f1-score']
    F1_test = cr_test['weighted avg']['f1-score']

    model_score = [precision_train, precision_test, recall_train, recall_test, acc_train, acc_test, F1_train, F1_test ]
    return model_score
```

# Creating DataFrame


```python
# Create a score dataframe
score=pd.DataFrame(index=['Precision Train', 'Precision Test','Recall Train','Recall Test','Accuracy Train', 'Accuracy Test', 'F1 macro Train', 'F1 macro Test'])
```

# ML Model - 1 : Logistic regression


```python
#Ml model -1 implementation
lr_model=LogisticRegression(fit_intercept=True,max_iter=1000)
# Model is trained (fit) and predicted in the evaluate model.
```

# 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.


```python
#Visualization evaluation metric score chart
lr_score= evaluate_model(lr_model,x_train,x_test,y_train,y_test)
```

    
    Confusion Matrix
    


    
![png](images/output_63_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 0.971429 |   0.985507 |     35    |
    | 1            |    1        | 1        |   1        |     19    |
    | 2            |    0.978723 | 1        |   0.989247 |     46    |
    | accuracy     |    0.99     | 0.99     |   0.99     |      0.99 |
    | macro avg    |    0.992908 | 0.990476 |   0.991585 |    100    |
    | weighted avg |    0.990213 | 0.99     |   0.989981 |    100    |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 0.971429 |   0.985507 |     35    |
    | 1            |    1        | 1        |   1        |     19    |
    | 2            |    0.978723 | 1        |   0.989247 |     46    |
    | accuracy     |    0.99     | 0.99     |   0.99     |      0.99 |
    | macro avg    |    0.992908 | 0.990476 |   0.991585 |    100    |
    | weighted avg |    0.990213 | 0.99     |   0.989981 |    100    |
    


```python
# Updated Evaluation metric Score Chart
score['Logistic_Regression'] = lr_score
score
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Logistic_Regression</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.990213</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.990000</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.990000</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.989981</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Cross- Validation & Hyperparameter Tuning


```python
#Ml model -1 implementation with hyperparameter optimization techniques
#(ii.e, GridSearchCV,RandomSearch CV, Bayesian Optimization etc.)
# Define the hyperparameter grid
param_grid={'C':[100,10,1,0,1,0.01,0.001,0.0001],
           'penalty':['11','12','elasticnet'],
            'l1_ratio':[0.5],'max_iter':[10000],
           'solver':['newton-cg','lbfgs','liblinear','sag','saga']}

#initialization the logistic regression model.
logreg=LogisticRegression(fit_intercept=True,max_iter=10000,random_state=0)

#Repeated Stratified kfold
rskf=RepeatedStratifiedKFold(n_splits=3,n_repeats=4,random_state=0)

#using GridSearchCV tune the hyperparameter using cross-validation
grid= GridSearchCV(logreg,param_grid,cv=rskf)
grid.fit(x_train,y_train)

#Select the best hyperparameters found by GridSearchCV
best_params=grid.best_params_
print("Best Hyperparameters:",best_params)
```

    Best Hyperparameters: {'C': 100, 'l1_ratio': 0.5, 'max_iter': 10000, 'penalty': 'elasticnet', 'solver': 'saga'}
    


```python
#initiate model with best parameters
lr_model2=LogisticRegression(C=best_params['C'],
                             penalty=best_params['penalty'],
                             l1_ratio=best_params['l1_ratio'],
                            solver=best_params['solver'],
                            max_iter=10000,random_state=0)
```


```python
# Visualizing evaluation Metric Score chart
lr_score2 = evaluate_model(lr_model2, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix
    


    
![png](images/output_68_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    0.790698 | 0.971429 |   0.871795 |      35   |
    | 1            |    1        | 0.894737 |   0.944444 |      19   |
    | 2            |    0.975    | 0.847826 |   0.906977 |      46   |
    | accuracy     |    0.9      | 0.9      |   0.9      |       0.9 |
    | macro avg    |    0.921899 | 0.904664 |   0.907739 |     100   |
    | weighted avg |    0.915244 | 0.9      |   0.901782 |     100   |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    0.790698 | 0.971429 |   0.871795 |      35   |
    | 1            |    1        | 0.894737 |   0.944444 |      19   |
    | 2            |    0.975    | 0.847826 |   0.906977 |      46   |
    | accuracy     |    0.9      | 0.9      |   0.9      |       0.9 |
    | macro avg    |    0.921899 | 0.904664 |   0.907739 |     100   |
    | weighted avg |    0.915244 | 0.9      |   0.901782 |     100   |
    


```python
score['logistic_regression tuned'] = lr_score2
# Updated Evaluation metric Score Chart
score
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Logistic_Regression</th>
      <th>logistic_regression tuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>1.000000</td>
      <td>0.825458</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.990213</td>
      <td>0.915244</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>1.000000</td>
      <td>0.795855</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.989981</td>
      <td>0.901782</td>
    </tr>
  </tbody>
</table>
</div>



# ML MOdel -2 Decision Tree


```python
# ML Model-2 Implementationdt
dt_model=DecisionTreeClassifier(random_state=20)
#Model is trained (fit) and predicted in the 
```

# 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.


```python
# Visualizing evaluation Metric Score chart
dt_score = evaluate_model(dt_model, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix
    


    
![png](images/output_73_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    0.942857 | 0.942857 |   0.942857 |     35    |
    | 1            |    0.9      | 0.947368 |   0.923077 |     19    |
    | 2            |    0.977778 | 0.956522 |   0.967033 |     46    |
    | accuracy     |    0.95     | 0.95     |   0.95     |      0.95 |
    | macro avg    |    0.940212 | 0.948916 |   0.944322 |    100    |
    | weighted avg |    0.950778 | 0.95     |   0.95022  |    100    |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    0.942857 | 0.942857 |   0.942857 |     35    |
    | 1            |    0.9      | 0.947368 |   0.923077 |     19    |
    | 2            |    0.977778 | 0.956522 |   0.967033 |     46    |
    | accuracy     |    0.95     | 0.95     |   0.95     |      0.95 |
    | macro avg    |    0.940212 | 0.948916 |   0.944322 |    100    |
    | weighted avg |    0.950778 | 0.95     |   0.95022  |    100    |
    


```python
# Updated Evaluation metric Score Chart
score['decision_tree'] = dt_score
score
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Logistic_Regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>1.000000</td>
      <td>0.825458</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.990213</td>
      <td>0.915244</td>
      <td>0.950778</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>1.000000</td>
      <td>0.795855</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.989981</td>
      <td>0.901782</td>
      <td>0.950220</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Cross Validation & Hyperparameter Tuning


```python
# ML Model - 2 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
# Define the hyperparameter grid
grid = {'max_depth' : [3,4,5,6,7,8],
        'min_samples_split' : np.arange(2,8),
        'min_samples_leaf' : np.arange(10,20)}

# Initialize the model
model = DecisionTreeClassifier()

# repeated stratified kfold
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=0)

# Initialize GridSearchCV
grid_search = GridSearchCV(model, grid, cv=rskf)

# Fit the GridSearchCV to the training data
grid_search.fit(x_train, y_train)

# Select the best hyperparameters
best_params = grid_search.best_params_
print("best_hyperparameters: ", best_params)
```

    best_hyperparameters:  {'max_depth': 3, 'min_samples_leaf': 10, 'min_samples_split': 2}
    


```python
#Train a new model with the best hyperparameters
dt_model2 = DecisionTreeClassifier(max_depth=best_params['max_depth'],
                                 min_samples_leaf=best_params['min_samples_leaf'],
                                 min_samples_split=best_params['min_samples_split'],
                                 random_state=20)
                                  
# Visualizing evaluation Metric Score chart
dt2_score = evaluate_model(dt_model2, x_train, x_test, y_train, y_test)

score['decision_tree_tuned'] = dt2_score
# Updated Evaluation metric Score Chart
score
```

    
    Confusion Matrix
    


    
![png](images/output_77_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    0.853659 | 1        |   0.921053 |     35    |
    | 1            |    0.761905 | 0.842105 |   0.8      |     19    |
    | 2            |    0.973684 | 0.804348 |   0.880952 |     46    |
    | accuracy     |    0.88     | 0.88     |   0.88     |      0.88 |
    | macro avg    |    0.863083 | 0.882151 |   0.867335 |    100    |
    | weighted avg |    0.891437 | 0.88     |   0.879607 |    100    |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    0.853659 | 1        |   0.921053 |     35    |
    | 1            |    0.761905 | 0.842105 |   0.8      |     19    |
    | 2            |    0.973684 | 0.804348 |   0.880952 |     46    |
    | accuracy     |    0.88     | 0.88     |   0.88     |      0.88 |
    | macro avg    |    0.863083 | 0.882151 |   0.867335 |    100    |
    | weighted avg |    0.891437 | 0.88     |   0.879607 |    100    |
    




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Logistic_Regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>1.000000</td>
      <td>0.825458</td>
      <td>1.000000</td>
      <td>0.939333</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.990213</td>
      <td>0.915244</td>
      <td>0.950778</td>
      <td>0.891437</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>1.000000</td>
      <td>0.795855</td>
      <td>1.000000</td>
      <td>0.936242</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.989981</td>
      <td>0.901782</td>
      <td>0.950220</td>
      <td>0.879607</td>
    </tr>
  </tbody>
</table>
</div>



# ML Model - 3 : Random Forest


```python
# ML Model - 3 Implementation
rf_model = RandomForestClassifier(random_state=0)
# Model is trained (fit) and predicted in the evaluate model
```

# 1. Explain the ML Model used and it's performance using Evaluation Metric Score Chart.


```python
#visualization evaluation metric score chart
rf_score = evaluate_model(rf_model, x_train, x_test, y_train, y_test)

# Updated Evaluation metric Score Chart
score['random_forest'] = rf_score
score
```

    
    Confusion Matrix
    


    
![png](images/output_81_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |           1 |        1 |          1 |        35 |
    | 1            |           1 |        1 |          1 |        19 |
    | 2            |           1 |        1 |          1 |        46 |
    | accuracy     |           1 |        1 |          1 |         1 |
    | macro avg    |           1 |        1 |          1 |       100 |
    | weighted avg |           1 |        1 |          1 |       100 |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |           1 |        1 |          1 |        35 |
    | 1            |           1 |        1 |          1 |        19 |
    | 2            |           1 |        1 |          1 |        46 |
    | accuracy     |           1 |        1 |          1 |         1 |
    | macro avg    |           1 |        1 |          1 |       100 |
    | weighted avg |           1 |        1 |          1 |       100 |
    



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Logistic_Regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>1.000000</td>
      <td>0.825458</td>
      <td>1.000000</td>
      <td>0.939333</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.990213</td>
      <td>0.915244</td>
      <td>0.950778</td>
      <td>0.891437</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>1.000000</td>
      <td>0.795855</td>
      <td>1.000000</td>
      <td>0.936242</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.989981</td>
      <td>0.901782</td>
      <td>0.950220</td>
      <td>0.879607</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Cross- Validation & Hyperparameter Tuning


```python
# ML Model - 3 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
# Define the hyperparameter grid
grid = {'n_estimators': [10, 50, 100, 200],
              'max_depth': [8, 9, 10, 11, 12,13, 14, 15],
              'min_samples_split': [2, 3, 4, 5]}

# Initialize the model
rf = RandomForestClassifier(random_state=0)

# Repeated stratified kfold
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=0)

# Initialize RandomSearchCV
random_search = RandomizedSearchCV(rf, grid,cv=rskf, n_iter=10, n_jobs=-1)

# Fit the RandomSearchCV to the training data
random_search.fit(x_train, y_train)

# Select the best hyperparameters
best_params = random_search.best_params_
print("best_hyperparameters: ", best_params)

# Initialize model with best parameters
rf_model2 = RandomForestClassifier(n_estimators = best_params['n_estimators'],
                                 min_samples_leaf= best_params['min_samples_split'],
                                 max_depth = best_params['max_depth'],
                                 random_state=0)

# Visualizing evaluation Metric Score chart
rf2_score = evaluate_model(rf_model2, x_train, x_test, y_train, y_test)

score['random_forest_tuned'] = rf2_score
#Updated Evaluation metric Score Chart
score

```

    best_hyperparameters:  {'n_estimators': 200, 'min_samples_split': 4, 'max_depth': 11}
    
    Confusion Matrix
    


    
![png](images/output_83_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    0.945946 | 1        |   0.972222 |     35    |
    | 1            |    1        | 0.894737 |   0.944444 |     19    |
    | 2            |    1        | 1        |   1        |     46    |
    | accuracy     |    0.98     | 0.98     |   0.98     |      0.98 |
    | macro avg    |    0.981982 | 0.964912 |   0.972222 |    100    |
    | weighted avg |    0.981081 | 0.98     |   0.979722 |    100    |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    0.945946 | 1        |   0.972222 |     35    |
    | 1            |    1        | 0.894737 |   0.944444 |     19    |
    | 2            |    1        | 1        |   1        |     46    |
    | accuracy     |    0.98     | 0.98     |   0.98     |      0.98 |
    | macro avg    |    0.981982 | 0.964912 |   0.972222 |    100    |
    | weighted avg |    0.981081 | 0.98     |   0.979722 |    100    |
    




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Logistic_Regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>1.000000</td>
      <td>0.825458</td>
      <td>1.000000</td>
      <td>0.939333</td>
      <td>1.0</td>
      <td>0.978477</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.990213</td>
      <td>0.915244</td>
      <td>0.950778</td>
      <td>0.891437</td>
      <td>1.0</td>
      <td>0.981081</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>1.000000</td>
      <td>0.795855</td>
      <td>1.000000</td>
      <td>0.936242</td>
      <td>1.0</td>
      <td>0.978478</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.989981</td>
      <td>0.901782</td>
      <td>0.950220</td>
      <td>0.879607</td>
      <td>1.0</td>
      <td>0.979722</td>
    </tr>
  </tbody>
</table>
</div>



# ML Model - 4 : SVM (Support Vector Machine)


```python
# ML Model - 4 Implementation
svm_model = SVC(kernel='linear', random_state=0, probability=True)
# Model is trained (fit) and predicted in the evaluate model
```

# 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.


```python
# Visualizing evaluation Metric Score chart
svm_score = evaluate_model(svm_model, x_train, x_test, y_train, y_test)
#Updated Evaluation metric Score Chart
score['s_v_m'] = svm_score
score
```

    
    Confusion Matrix
    


    
![png](images/output_87_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |           1 |        1 |          1 |        35 |
    | 1            |           1 |        1 |          1 |        19 |
    | 2            |           1 |        1 |          1 |        46 |
    | accuracy     |           1 |        1 |          1 |         1 |
    | macro avg    |           1 |        1 |          1 |       100 |
    | weighted avg |           1 |        1 |          1 |       100 |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |           1 |        1 |          1 |        35 |
    | 1            |           1 |        1 |          1 |        19 |
    | 2            |           1 |        1 |          1 |        46 |
    | accuracy     |           1 |        1 |          1 |         1 |
    | macro avg    |           1 |        1 |          1 |       100 |
    | weighted avg |           1 |        1 |          1 |       100 |
    


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Logistic_Regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
      <th>s_v_m</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>1.000000</td>
      <td>0.825458</td>
      <td>1.000000</td>
      <td>0.939333</td>
      <td>1.0</td>
      <td>0.978477</td>
      <td>0.991416</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.990213</td>
      <td>0.915244</td>
      <td>0.950778</td>
      <td>0.891437</td>
      <td>1.0</td>
      <td>0.981081</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
      <td>0.991416</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
      <td>0.991416</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>1.000000</td>
      <td>0.795855</td>
      <td>1.000000</td>
      <td>0.936242</td>
      <td>1.0</td>
      <td>0.978478</td>
      <td>0.991416</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.989981</td>
      <td>0.901782</td>
      <td>0.950220</td>
      <td>0.879607</td>
      <td>1.0</td>
      <td>0.979722</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



# ML Model - 5 : Xtreme Gradient Boosting


```python
# ML Model - 5 Implementation
xgb_model = xgb.XGBClassifier()
# Model is trained (fit) and predicted in the evaluate model
```

# 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.


```python
#Visualizing evaluation Metric Score chart
xgb_score = evaluate_model(xgb_model, x_train, x_test, y_train, y_test)
# Updated Evaluation metric Score Chart
score['x_g_b'] = xgb_score
score
```

    
    Confusion Matrix
    


    
![png](images/output_91_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        |     35    |
    | 1            |    0.95     | 1        |   0.974359 |     19    |
    | 2            |    1        | 0.978261 |   0.989011 |     46    |
    | accuracy     |    0.99     | 0.99     |   0.99     |      0.99 |
    | macro avg    |    0.983333 | 0.992754 |   0.98779  |    100    |
    | weighted avg |    0.9905   | 0.99     |   0.990073 |    100    |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        |     35    |
    | 1            |    0.95     | 1        |   0.974359 |     19    |
    | 2            |    1        | 0.978261 |   0.989011 |     46    |
    | accuracy     |    0.99     | 0.99     |   0.99     |      0.99 |
    | macro avg    |    0.983333 | 0.992754 |   0.98779  |    100    |
    | weighted avg |    0.9905   | 0.99     |   0.990073 |    100    |
    




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Logistic_Regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
      <th>s_v_m</th>
      <th>x_g_b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>1.000000</td>
      <td>0.825458</td>
      <td>1.000000</td>
      <td>0.939333</td>
      <td>1.0</td>
      <td>0.978477</td>
      <td>0.991416</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.990213</td>
      <td>0.915244</td>
      <td>0.950778</td>
      <td>0.891437</td>
      <td>1.0</td>
      <td>0.981081</td>
      <td>1.000000</td>
      <td>0.990500</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
      <td>0.991416</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
      <td>1.000000</td>
      <td>0.990000</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
      <td>0.991416</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
      <td>1.000000</td>
      <td>0.990000</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>1.000000</td>
      <td>0.795855</td>
      <td>1.000000</td>
      <td>0.936242</td>
      <td>1.0</td>
      <td>0.978478</td>
      <td>0.991416</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.989981</td>
      <td>0.901782</td>
      <td>0.950220</td>
      <td>0.879607</td>
      <td>1.0</td>
      <td>0.979722</td>
      <td>1.000000</td>
      <td>0.990073</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Cross- Validation & Hyperparameter Tuning


```python
# ML Model - 5 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
# Define the hyperparameter grid
param_grid = {'learning_rate': np.arange(0.01, 0.3, 0.01),
              'max_depth': np.arange(3, 15, 1),
              'n_estimators': np.arange(100, 200, 10)}

# Initialize the model
xgb2 = xgb.XGBClassifier(random_state=0)

# Repeated stratified kfold
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=0)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(xgb2, param_grid, n_iter=10, cv=rskf)

# Fit the RandomizedSearchCV to the training data
random_search.fit(x_train, y_train)

# Select the best hyperparameters
best_params = random_search.best_params_
print("best_hyperparameters: ", best_params)

# Initialize model with best parameters
xgb_model2 = xgb.XGBClassifier(learning_rate = best_params['learning_rate'],
                                 max_depth = best_params['max_depth'],
                               n_estimators = best_params['n_estimators'],
                                 random_state=0)

# Visualizing evaluation Metric Score chart
xgb2_score = evaluate_model(xgb_model2, x_train, x_test, y_train, y_test)

score['x_g_b_tuned'] = xgb2_score
# Updated Evaluation metric Score Chart
score
```

    best_hyperparameters:  {'n_estimators': 160, 'max_depth': 13, 'learning_rate': 0.11}
    
    Confusion Matrix
    


    
![png](images/output_93_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    0.972222 | 1        |   0.985915 |     35    |
    | 1            |    0.95     | 1        |   0.974359 |     19    |
    | 2            |    1        | 0.956522 |   0.977778 |     46    |
    | accuracy     |    0.98     | 0.98     |   0.98     |      0.98 |
    | macro avg    |    0.974074 | 0.985507 |   0.979351 |    100    |
    | weighted avg |    0.980778 | 0.98     |   0.979976 |    100    |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    0.972222 | 1        |   0.985915 |     35    |
    | 1            |    0.95     | 1        |   0.974359 |     19    |
    | 2            |    1        | 0.956522 |   0.977778 |     46    |
    | accuracy     |    0.98     | 0.98     |   0.98     |      0.98 |
    | macro avg    |    0.974074 | 0.985507 |   0.979351 |    100    |
    | weighted avg |    0.980778 | 0.98     |   0.979976 |    100    |
    



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Logistic_Regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
      <th>s_v_m</th>
      <th>x_g_b</th>
      <th>x_g_b_tuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>1.000000</td>
      <td>0.825458</td>
      <td>1.000000</td>
      <td>0.939333</td>
      <td>1.0</td>
      <td>0.978477</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.990213</td>
      <td>0.915244</td>
      <td>0.950778</td>
      <td>0.891437</td>
      <td>1.0</td>
      <td>0.981081</td>
      <td>1.000000</td>
      <td>0.990500</td>
      <td>0.980778</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>0.980000</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>0.980000</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>1.000000</td>
      <td>0.795855</td>
      <td>1.000000</td>
      <td>0.936242</td>
      <td>1.0</td>
      <td>0.978478</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.989981</td>
      <td>0.901782</td>
      <td>0.950220</td>
      <td>0.879607</td>
      <td>1.0</td>
      <td>0.979722</td>
      <td>1.000000</td>
      <td>0.990073</td>
      <td>0.979976</td>
    </tr>
  </tbody>
</table>
</div>



# ML Model - 6 : Naive Bayes


```python
# Ml Model - 6 Implementation 
nb_model=GaussianNB()
# Model is trained (fit) and predicted in the evaluate model 

#Visualization evaluation Metric Score Chart 
nb_score=evaluate_model(nb_model,x_train,x_test,y_train,y_test)

# Updated Evaluation metric Score Chart 
score['naive_bayes']=nb_score
score
```

    
    Confusion Matrix
    


    
![png](images/output_95_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    0.945946 | 1        |   0.972222 |     35    |
    | 1            |    1        | 0.894737 |   0.944444 |     19    |
    | 2            |    1        | 1        |   1        |     46    |
    | accuracy     |    0.98     | 0.98     |   0.98     |      0.98 |
    | macro avg    |    0.981982 | 0.964912 |   0.972222 |    100    |
    | weighted avg |    0.981081 | 0.98     |   0.979722 |    100    |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    0.945946 | 1        |   0.972222 |     35    |
    | 1            |    1        | 0.894737 |   0.944444 |     19    |
    | 2            |    1        | 1        |   1        |     46    |
    | accuracy     |    0.98     | 0.98     |   0.98     |      0.98 |
    | macro avg    |    0.981982 | 0.964912 |   0.972222 |    100    |
    | weighted avg |    0.981081 | 0.98     |   0.979722 |    100    |
    

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Logistic_Regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
      <th>s_v_m</th>
      <th>x_g_b</th>
      <th>x_g_b_tuned</th>
      <th>naive_bayes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>1.000000</td>
      <td>0.825458</td>
      <td>1.000000</td>
      <td>0.939333</td>
      <td>1.0</td>
      <td>0.978477</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.970183</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.990213</td>
      <td>0.915244</td>
      <td>0.950778</td>
      <td>0.891437</td>
      <td>1.0</td>
      <td>0.981081</td>
      <td>1.000000</td>
      <td>0.990500</td>
      <td>0.980778</td>
      <td>0.981081</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.969957</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>0.980000</td>
      <td>0.980000</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.969957</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>0.980000</td>
      <td>0.980000</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>1.000000</td>
      <td>0.795855</td>
      <td>1.000000</td>
      <td>0.936242</td>
      <td>1.0</td>
      <td>0.978478</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.970041</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.989981</td>
      <td>0.901782</td>
      <td>0.950220</td>
      <td>0.879607</td>
      <td>1.0</td>
      <td>0.979722</td>
      <td>1.000000</td>
      <td>0.990073</td>
      <td>0.979976</td>
      <td>0.979722</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Cross- Validation & Hyperparameter Tuning


```python
# ML Model - 6 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
# Define the hyperparameter grid
param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}

# Initialize the model
naive = GaussianNB()

# repeated stratified kfold
rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=4, random_state=0)

# Initialize GridSearchCV
GridSearch = GridSearchCV(naive, param_grid, cv=rskf, n_jobs=-1)

# Fit the GridSearchCV to the training data
GridSearch.fit(x_train, y_train)

# Select the best hyperparameters
best_params = GridSearch.best_params_
print("best_hyperparameters: ", best_params)

# Initiate model with best parameters
nb_model2 = GaussianNB(var_smoothing = best_params['var_smoothing'])

# Visualizing evaluation Metric Score chart
nb2_score = evaluate_model(nb_model2, x_train, x_test, y_train, y_test)

score['naive_bayes_tuned']= nb2_score
# Updated Evaluation metric Score Chart
score
```

    best_hyperparameters:  {'var_smoothing': 2.310129700083158e-07}
    
    Confusion Matrix
    


    
![png](images/output_97_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    0.945946 | 1        |   0.972222 |     35    |
    | 1            |    1        | 0.894737 |   0.944444 |     19    |
    | 2            |    1        | 1        |   1        |     46    |
    | accuracy     |    0.98     | 0.98     |   0.98     |      0.98 |
    | macro avg    |    0.981982 | 0.964912 |   0.972222 |    100    |
    | weighted avg |    0.981081 | 0.98     |   0.979722 |    100    |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    0.945946 | 1        |   0.972222 |     35    |
    | 1            |    1        | 0.894737 |   0.944444 |     19    |
    | 2            |    1        | 1        |   1        |     46    |
    | accuracy     |    0.98     | 0.98     |   0.98     |      0.98 |
    | macro avg    |    0.981982 | 0.964912 |   0.972222 |    100    |
    | weighted avg |    0.981081 | 0.98     |   0.979722 |    100    |
    



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Logistic_Regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
      <th>s_v_m</th>
      <th>x_g_b</th>
      <th>x_g_b_tuned</th>
      <th>naive_bayes</th>
      <th>naive_bayes_tuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>1.000000</td>
      <td>0.825458</td>
      <td>1.000000</td>
      <td>0.939333</td>
      <td>1.0</td>
      <td>0.978477</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.970183</td>
      <td>0.970183</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.990213</td>
      <td>0.915244</td>
      <td>0.950778</td>
      <td>0.891437</td>
      <td>1.0</td>
      <td>0.981081</td>
      <td>1.000000</td>
      <td>0.990500</td>
      <td>0.980778</td>
      <td>0.981081</td>
      <td>0.981081</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.969957</td>
      <td>0.969957</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>0.980000</td>
      <td>0.980000</td>
      <td>0.980000</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.969957</td>
      <td>0.969957</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>0.980000</td>
      <td>0.980000</td>
      <td>0.980000</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>1.000000</td>
      <td>0.795855</td>
      <td>1.000000</td>
      <td>0.936242</td>
      <td>1.0</td>
      <td>0.978478</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.970041</td>
      <td>0.970041</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.989981</td>
      <td>0.901782</td>
      <td>0.950220</td>
      <td>0.879607</td>
      <td>1.0</td>
      <td>0.979722</td>
      <td>1.000000</td>
      <td>0.990073</td>
      <td>0.979976</td>
      <td>0.979722</td>
      <td>0.979722</td>
    </tr>
  </tbody>
</table>
</div>



# ML Model-7 Neural Network 


```python
# ML Model - 7 Implementation
nn_model = MLPClassifier(random_state=0)
# Model is trained (fit) and predicted in the evaluate model
```

# 1 Explain the ML Model used and its performance using Evaluation Metric Score Chart.


```python
# Visualizing evaluation Metric Score chart
neural_score = evaluate_model(nn_model, x_train, x_test, y_train, y_test)

# Updated Evaluation metric Score Chart
score['neural_network'] = neural_score
score
```

    
    Confusion Matrix
    


    
![png](images/output_101_1.png)
    


    
    Train Classification Report:
    |              |   precision |    recall |   f1-score |   support |
    |:-------------|------------:|----------:|-----------:|----------:|
    | 0            |    0.5      | 0.0571429 |   0.102564 |     35    |
    | 1            |    0.533333 | 0.421053  |   0.470588 |     19    |
    | 2            |    0.567901 | 1         |   0.724409 |     46    |
    | accuracy     |    0.56     | 0.56      |   0.56     |      0.56 |
    | macro avg    |    0.533745 | 0.492732  |   0.432521 |    100    |
    | weighted avg |    0.537568 | 0.56      |   0.458538 |    100    |
    
    Test Classification Report:
    |              |   precision |    recall |   f1-score |   support |
    |:-------------|------------:|----------:|-----------:|----------:|
    | 0            |    0.5      | 0.0571429 |   0.102564 |     35    |
    | 1            |    0.533333 | 0.421053  |   0.470588 |     19    |
    | 2            |    0.567901 | 1         |   0.724409 |     46    |
    | accuracy     |    0.56     | 0.56      |   0.56     |      0.56 |
    | macro avg    |    0.533745 | 0.492732  |   0.432521 |    100    |
    | weighted avg |    0.537568 | 0.56      |   0.458538 |    100    |
    

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Logistic_Regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
      <th>s_v_m</th>
      <th>x_g_b</th>
      <th>x_g_b_tuned</th>
      <th>naive_bayes</th>
      <th>naive_bayes_tuned</th>
      <th>neural_network</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>1.000000</td>
      <td>0.825458</td>
      <td>1.000000</td>
      <td>0.939333</td>
      <td>1.0</td>
      <td>0.978477</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.970183</td>
      <td>0.970183</td>
      <td>0.458453</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.990213</td>
      <td>0.915244</td>
      <td>0.950778</td>
      <td>0.891437</td>
      <td>1.0</td>
      <td>0.981081</td>
      <td>1.000000</td>
      <td>0.990500</td>
      <td>0.980778</td>
      <td>0.981081</td>
      <td>0.981081</td>
      <td>0.537568</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.969957</td>
      <td>0.969957</td>
      <td>0.394850</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>0.980000</td>
      <td>0.980000</td>
      <td>0.980000</td>
      <td>0.560000</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.969957</td>
      <td>0.969957</td>
      <td>0.394850</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>0.980000</td>
      <td>0.980000</td>
      <td>0.980000</td>
      <td>0.560000</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>1.000000</td>
      <td>0.795855</td>
      <td>1.000000</td>
      <td>0.936242</td>
      <td>1.0</td>
      <td>0.978478</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.970041</td>
      <td>0.970041</td>
      <td>0.301940</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.989981</td>
      <td>0.901782</td>
      <td>0.950220</td>
      <td>0.879607</td>
      <td>1.0</td>
      <td>0.979722</td>
      <td>1.000000</td>
      <td>0.990073</td>
      <td>0.979976</td>
      <td>0.979722</td>
      <td>0.979722</td>
      <td>0.458538</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Cross- Validation & Hyperparameter Tuning


```python
# ML Model - 7 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
# Define the hyperparameter grid
param_grid = {'hidden_layer_sizes': np.arange(10, 100, 10),
              'alpha': np.arange(0.0001, 0.01, 0.0001)}

# Initialize the model
neural = MLPClassifier(random_state=0)

# Repeated stratified kfold
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=0)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(neural, param_grid, n_iter=10, cv=rskf, n_jobs=-1)

# Fit the RandomizedSearchCV to the training data
random_search.fit(x_train, y_train)

# Select the best hyperparameters
best_params = random_search.best_params_
print("best_hyperparameters: ", best_params)

# Initiate model with best hyperparameters
nn_model2=MLPClassifier(hidden_layer_sizes=best_params['hidden_layer_sizes'],
                      alpha=best_params['alpha'],
                      random_state =0)
# Visualization Evaluation Metric Score Chart
neural2_score=evaluate_model(nn_model,x_train,x_test,y_train,y_test)

score['neural_network_tuned']=neural2_score
#Updated Evaluation Metric Score Chart
score
```

    best_hyperparameters:  {'hidden_layer_sizes': 40, 'alpha': 0.0071}
    
    Confusion Matrix
    


    
![png](images/output_103_1.png)
    


    
    Train Classification Report:
    |              |   precision |    recall |   f1-score |   support |
    |:-------------|------------:|----------:|-----------:|----------:|
    | 0            |    0.5      | 0.0571429 |   0.102564 |     35    |
    | 1            |    0.533333 | 0.421053  |   0.470588 |     19    |
    | 2            |    0.567901 | 1         |   0.724409 |     46    |
    | accuracy     |    0.56     | 0.56      |   0.56     |      0.56 |
    | macro avg    |    0.533745 | 0.492732  |   0.432521 |    100    |
    | weighted avg |    0.537568 | 0.56      |   0.458538 |    100    |
    
    Test Classification Report:
    |              |   precision |    recall |   f1-score |   support |
    |:-------------|------------:|----------:|-----------:|----------:|
    | 0            |    0.5      | 0.0571429 |   0.102564 |     35    |
    | 1            |    0.533333 | 0.421053  |   0.470588 |     19    |
    | 2            |    0.567901 | 1         |   0.724409 |     46    |
    | accuracy     |    0.56     | 0.56      |   0.56     |      0.56 |
    | macro avg    |    0.533745 | 0.492732  |   0.432521 |    100    |
    | weighted avg |    0.537568 | 0.56      |   0.458538 |    100    |
    




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Logistic_Regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
      <th>s_v_m</th>
      <th>x_g_b</th>
      <th>x_g_b_tuned</th>
      <th>naive_bayes</th>
      <th>naive_bayes_tuned</th>
      <th>neural_network</th>
      <th>neural_network_tuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>1.000000</td>
      <td>0.825458</td>
      <td>1.000000</td>
      <td>0.939333</td>
      <td>1.0</td>
      <td>0.978477</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.970183</td>
      <td>0.970183</td>
      <td>0.458453</td>
      <td>0.458453</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.990213</td>
      <td>0.915244</td>
      <td>0.950778</td>
      <td>0.891437</td>
      <td>1.0</td>
      <td>0.981081</td>
      <td>1.000000</td>
      <td>0.990500</td>
      <td>0.980778</td>
      <td>0.981081</td>
      <td>0.981081</td>
      <td>0.537568</td>
      <td>0.537568</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.969957</td>
      <td>0.969957</td>
      <td>0.394850</td>
      <td>0.394850</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>0.980000</td>
      <td>0.980000</td>
      <td>0.980000</td>
      <td>0.560000</td>
      <td>0.560000</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>1.000000</td>
      <td>0.802575</td>
      <td>1.000000</td>
      <td>0.935622</td>
      <td>1.0</td>
      <td>0.978541</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.969957</td>
      <td>0.969957</td>
      <td>0.394850</td>
      <td>0.394850</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.990000</td>
      <td>0.900000</td>
      <td>0.950000</td>
      <td>0.880000</td>
      <td>1.0</td>
      <td>0.980000</td>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>0.980000</td>
      <td>0.980000</td>
      <td>0.980000</td>
      <td>0.560000</td>
      <td>0.560000</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>1.000000</td>
      <td>0.795855</td>
      <td>1.000000</td>
      <td>0.936242</td>
      <td>1.0</td>
      <td>0.978478</td>
      <td>0.991416</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.970041</td>
      <td>0.970041</td>
      <td>0.301940</td>
      <td>0.301940</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.989981</td>
      <td>0.901782</td>
      <td>0.950220</td>
      <td>0.879607</td>
      <td>1.0</td>
      <td>0.979722</td>
      <td>1.000000</td>
      <td>0.990073</td>
      <td>0.979976</td>
      <td>0.979722</td>
      <td>0.979722</td>
      <td>0.458538</td>
      <td>0.458538</td>
    </tr>
  </tbody>
</table>
</div>



# **MarkDown**


```python
print(score.to_markdown())
```

    |                 |   Logistic_Regression |   logistic_regression tuned |   decision_tree |   decision_tree_tuned |   random_forest |   random_forest_tuned |    s_v_m |    x_g_b |   x_g_b_tuned |   naive_bayes |   naive_bayes_tuned |   neural_network |   neural_network_tuned |
    |:----------------|----------------------:|----------------------------:|----------------:|----------------------:|----------------:|----------------------:|---------:|---------:|--------------:|--------------:|--------------------:|-----------------:|-----------------------:|
    | Precision Train |              1        |                    0.825458 |        1        |              0.939333 |               1 |              0.978477 | 0.991416 | 1        |      1        |      0.970183 |            0.970183 |         0.458453 |               0.458453 |
    | Precision Test  |              0.990213 |                    0.915244 |        0.950778 |              0.891437 |               1 |              0.981081 | 1        | 0.9905   |      0.980778 |      0.981081 |            0.981081 |         0.537568 |               0.537568 |
    | Recall Train    |              1        |                    0.802575 |        1        |              0.935622 |               1 |              0.978541 | 0.991416 | 1        |      1        |      0.969957 |            0.969957 |         0.39485  |               0.39485  |
    | Recall Test     |              0.99     |                    0.9      |        0.95     |              0.88     |               1 |              0.98     | 1        | 0.99     |      0.98     |      0.98     |            0.98     |         0.56     |               0.56     |
    | Accuracy Train  |              1        |                    0.802575 |        1        |              0.935622 |               1 |              0.978541 | 0.991416 | 1        |      1        |      0.969957 |            0.969957 |         0.39485  |               0.39485  |
    | Accuracy Test   |              0.99     |                    0.9      |        0.95     |              0.88     |               1 |              0.98     | 1        | 0.99     |      0.98     |      0.98     |            0.98     |         0.56     |               0.56     |
    | F1 macro Train  |              1        |                    0.795855 |        1        |              0.936242 |               1 |              0.978478 | 0.991416 | 1        |      1        |      0.970041 |            0.970041 |         0.30194  |               0.30194  |
    | F1 macro Test   |              0.989981 |                    0.901782 |        0.95022  |              0.879607 |               1 |              0.979722 | 1        | 0.990073 |      0.979976 |      0.979722 |            0.979722 |         0.458538 |               0.458538 |
    

# Selection Of Best Model


```python
# Removing the overfitted models which have precision, recall, f1 scores for train as 1
score_t = score.transpose()            # taking transpose of the score dataframe to create new difference column
remove_models = score_t[score_t['Recall Train']>=0.98].index  # creating a list of models which have 1 for train and score_t['Accuracy Train']==1.0 and score_t['Precision Train']==1.0 and score_t['F1 macro Train']==1.0
remove_models

abc = score_t.drop(remove_models)                     # creating a new dataframe with required models
abc
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision Train</th>
      <th>Precision Test</th>
      <th>Recall Train</th>
      <th>Recall Test</th>
      <th>Accuracy Train</th>
      <th>Accuracy Test</th>
      <th>F1 macro Train</th>
      <th>F1 macro Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>logistic_regression tuned</th>
      <td>0.825458</td>
      <td>0.915244</td>
      <td>0.802575</td>
      <td>0.90</td>
      <td>0.802575</td>
      <td>0.90</td>
      <td>0.795855</td>
      <td>0.901782</td>
    </tr>
    <tr>
      <th>decision_tree_tuned</th>
      <td>0.939333</td>
      <td>0.891437</td>
      <td>0.935622</td>
      <td>0.88</td>
      <td>0.935622</td>
      <td>0.88</td>
      <td>0.936242</td>
      <td>0.879607</td>
    </tr>
    <tr>
      <th>random_forest_tuned</th>
      <td>0.978477</td>
      <td>0.981081</td>
      <td>0.978541</td>
      <td>0.98</td>
      <td>0.978541</td>
      <td>0.98</td>
      <td>0.978478</td>
      <td>0.979722</td>
    </tr>
    <tr>
      <th>naive_bayes</th>
      <td>0.970183</td>
      <td>0.981081</td>
      <td>0.969957</td>
      <td>0.98</td>
      <td>0.969957</td>
      <td>0.98</td>
      <td>0.970041</td>
      <td>0.979722</td>
    </tr>
    <tr>
      <th>naive_bayes_tuned</th>
      <td>0.970183</td>
      <td>0.981081</td>
      <td>0.969957</td>
      <td>0.98</td>
      <td>0.969957</td>
      <td>0.98</td>
      <td>0.970041</td>
      <td>0.979722</td>
    </tr>
    <tr>
      <th>neural_network</th>
      <td>0.458453</td>
      <td>0.537568</td>
      <td>0.394850</td>
      <td>0.56</td>
      <td>0.394850</td>
      <td>0.56</td>
      <td>0.301940</td>
      <td>0.458538</td>
    </tr>
    <tr>
      <th>neural_network_tuned</th>
      <td>0.458453</td>
      <td>0.537568</td>
      <td>0.394850</td>
      <td>0.56</td>
      <td>0.394850</td>
      <td>0.56</td>
      <td>0.301940</td>
      <td>0.458538</td>
    </tr>
  </tbody>
</table>
</div>




```python
def select_best_model(df, metrics):

    best_models = {}
    for metric in metrics:
        max_test = df[metric + ' Test'].max()
        best_model_test = df[df[metric + ' Test'] == max_test].index[0]
        best_model = best_model_test
        best_models[metric] = best_model
    return best_models

metrics = ['Precision', 'Recall', 'Accuracy', 'F1 macro']
best_models = select_best_model(abc, metrics)
print("The best models are:")
for metric, best_model in best_models.items():
    print(f"{metric}: {best_model} - {abc[metric+' Test'][best_model].round(4)}")

# Take recall as the primary evaluation metric
score_smpl = score.transpose()
remove_overfitting_models = score_smpl[score_smpl['Recall Train']>=0.98].index
remove_overfitting_models
new_score = score_smpl.drop(remove_overfitting_models)
new_score = new_score.drop(['Precision Train','Precision Test','Accuracy Train','Accuracy Test','F1 macro Train','F1 macro Test'], axis=1)
new_score.index.name = 'Classification Model'
print(new_score.to_markdown())
```

    The best models are:
    Precision: random_forest_tuned - 0.9811
    Recall: random_forest_tuned - 0.98
    Accuracy: random_forest_tuned - 0.98
    F1 macro: random_forest_tuned - 0.9797
    | Classification Model      |   Recall Train |   Recall Test |
    |:--------------------------|---------------:|--------------:|
    | logistic_regression tuned |       0.802575 |          0.9  |
    | decision_tree_tuned       |       0.935622 |          0.88 |
    | random_forest_tuned       |       0.978541 |          0.98 |
    | naive_bayes               |       0.969957 |          0.98 |
    | naive_bayes_tuned         |       0.969957 |          0.98 |
    | neural_network            |       0.39485  |          0.56 |
    | neural_network_tuned      |       0.39485  |          0.56 |
    

# Explain the model which have used for the prediction


```python
# Define a list of category labels for reference.
Category_RF = ['Adelie','Gentoo','Chinstrap']

# In this example, it's a data point with Sepal Length, Sepal Width, Petal Length, and Petal Width.
x_rf = np.array([[5.1, 3.5, 1.4, 0.2,0]])

# Use the tuned random forest model (rf_model2) to make a prediction.
x_rf_prediction = rf_model2.predict(x_rf)
x_rf_prediction[0]

# Convert the prediction to an integer and access the correct label from Category_RF
predicted_label = Category_RF[int(x_rf_prediction[0])]

# Display the predicted category label.
print(Category_RF[int(x_rf_prediction[0])])
```

    Adelie
    

# Testing Accuracy


```python
# Evaluate the model 
accuracy = rf_model.score(x_test, y_test)
print(f"Model Accuracy: {accuracy}")

```

    Model Accuracy: 1.0
    

# Saving iris_flower_model.pkl as file


```python
# After training, add this code to save the model:
import pickle
with open('penguin_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)  # Save the trained model to a file

# Once you run the cell with this code, 'penguin_model.pkl' will be created in your folder

```
