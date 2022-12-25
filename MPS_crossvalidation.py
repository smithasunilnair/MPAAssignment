
# # Estimation of Used Car Prices - Data Science project
# 

# ## Import Libraries
# All libraries are used for specific tasks including data preprocessing, visualization, transformation and evaluation

# Using cross validation technique - hence no test dataset separately




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor 
import warnings
warnings.filterwarnings("ignore")


# "--------------Business Problem: Estimate the price of a used car-----------"

# ## Import Data
# ### Read Training Data
# The training set is read locally and the **head** function is used to display the data for intial understanding

# "======Data understanding======"




dataTrain = pd.read_csv('data_train.csv')


# The **shape** function displays the number of rows and columns in the training set



dataTrain.shape


# Checking for null values in each column and displaying the sum of all null values in each column (Training Set)




dataTrain.isnull().sum()


# Checking for null values in each column and displaying the sum of all null values in each column (Testing Set)

# Removing the rows with empty values since the number of empty rows are small. This is the best approach compared to replacing with mean or random values




dataTrain=dataTrain.dropna()


# Checking if null values are eliminated (Training set)



dataTrain.isnull().sum()




dataTrain.shape


# Checking if null values are eliminated (Testing set)

# Checking the data types to see if all the data is in correct format. All the data seems to be in their required format.



dataTrain.dtypes


# Checking the correlation between the numerical features

# ## EDA (Exploratory Data Analysis)
# Visualizations are used to understand the relationship between the target variable and the features, in addition to correlation coefficient and p-value. 
# The visuals include heatmap, scatterplot,boxplot etc.
# 

# #Heat map




plt.figure(figsize=(10,6))
corr = dataTrain.corr()
sns.heatmap(corr,annot=True)
plt.show()


# It is clear from the box plot that the year_produced is the best discriminating feature

# ### Regression/scatter Plot
# This regression plot show the relation between **odometer** and **price**. A slight negative correlation is observed
# this shows that price is being affected by the change in odometer value.



plt.figure(figsize=(10,6))
sns.regplot(x="odometer_value", y="price_usd", data=dataTrain)


# As observed in the plot, a **negative correlation** is observed




from scipy import stats
pearson_coef, p_value = stats.pearsonr(dataTrain['odometer_value'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# Pearson corr coeff of -0.42 is obtained along with a p-value of 0. The p value confirms that the calculated correlation is **significant** hence this feature is significant to the prediction of used car price.

# The regression plot below shows a relationship between the year that the car is produced and the price of the car. A positive 
# correlation is observed between the two variables. This shows that the price increases with increase in production year of the car.




plt.figure(figsize=(10,6))
sns.regplot(x="year_produced", y="price_usd", data=dataTrain)


# As observed above, a high positive correlation of 0.7 is calculated along with the p-value of 0. This indicates that the correlation between the variables is significant hence year produced feature can be used for prediction.




pearson_coef, p_value = stats.pearsonr(dataTrain['year_produced'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# check for coorelation between 'engine_capacity' and 'price'




plt.figure(figsize=(10,6))
sns.regplot(x="engine_capacity", y="price_usd", data=dataTrain)


# A 0.3 correlation is calculated which is very small with a p value of 0. This indicates that even though the correlation is small but its 30% of 100 which is significant hence this feature can be used for predicition.




pearson_coef, p_value = stats.pearsonr(dataTrain['engine_capacity'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# This regression plot shows an minor positive correlation observed with the help of the best fit line. The calculation will confirm the actual value.

# -----check for correlation between 'number o fphotos' and 'price'------------




plt.figure(figsize=(10,6))
sns.regplot(x="number_of_photos", y="price_usd", data=dataTrain)


# The correlation is 0.31 based on the calculation while the p-value calculated is zero. This is similar to the last feature hence the significant 31% of 100 correlation makes this feature eligble for prediction.




pearson_coef, p_value = stats.pearsonr(dataTrain['number_of_photos'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# This plot shows correlation with points all over the graph like the previous feature varibale.

# -------check correlation b/w number of mantenance and price-------------




plt.figure(figsize=(10,6))
sns.regplot(x="number_of_maintenance", y="price_usd", data=dataTrain)


# The calculation proves that a correlation is lesser than 0.1 percent is same as no correlation and the p-value of lesser than 0.01 confirms it. This feature is not significant enough for predicition




pearson_coef, p_value = stats.pearsonr(dataTrain['number_of_maintenance'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# ---- this plot shows no correlation with points all over the graph since Pearson coeff is < 0.1 and p value is less than 0.01----

# *************check correlation between duration listed and price***************




plt.figure(figsize=(10,6))
sns.regplot(x="duration_listed", y="price_usd", data=dataTrain)


# The calculated correlation is lesser than 0.1 which is considered negligible. The p-value lesser than 0.01 confirming the correlation value hence this feature is not suitable for prediction of price. 




pearson_coef, p_value = stats.pearsonr(dataTrain['duration_listed'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# ### Box Plot
# These plots are used for categorical data to determine the importance of features for prediction. 

# In the given plot below, it is observed that the price range vary for automatic and manual transmisson. This indicates the categories can vary with price hence feature can be used for prediction




sns.boxplot(x="transmission", y="price_usd", data=dataTrain)


# The box plot shows how prices vary based on different colors. This shows that color can be used as a feature for price prediction.




plt.figure(figsize=(10,6))
sns.boxplot(x="color", y="price_usd", data=dataTrain)


# This plot shows engine fuel types and how they affect the price. Hybrid petroll with the highest price range while hybrid diesel with lowest price range. This feature can be used for prediction.




sns.boxplot(x="engine_fuel", y="price_usd", data=dataTrain)


# The engine type (based on fuel type) shows that both categories have almost the same price range which will not bring differences in price when prediction is made. Hence this feature is not suitable for price prediction




sns.boxplot(x="engine_type", y="price_usd", data=dataTrain)


# Thee box plot shows body type categories with varying prices per category hence this feature can be used for price prediction, not so signficant though




plt.figure(figsize=(10,6))
sns.boxplot(x="body_type", y="price_usd", data=dataTrain)


# Has warranty feature shows a huge difference in price ranges between cars with warrant and vice versa. This feature is very important for price prediction as the bigger the difference in range the better the feature.




sns.boxplot(x="has_warranty", y="price_usd", data=dataTrain)


# This feature is similar to the feature above, all three categories have wider price ranges between one another. This feature is also crucial for price prediction.




sns.boxplot(x="ownership", y="price_usd", data=dataTrain)


# Front and rear drive have **minimal price difference** while all drive shows a **greater difference** hence the feature can be used for prediction.




sns.boxplot(x="type_of_drive", y="price_usd", data=dataTrain)


# With not same price range between categories this feature is  suitable for prediction.




sns.boxplot(x="is_exchangeable", y="price_usd", data=dataTrain)


# This plot shows that the manufacturer name is not important when selling a car. The variety of price ranges for all categories prove that the feature is insignificant for price prediction.




plt.figure(figsize=(10,6))
sns.boxplot(x="manufacturer_name", y="price_usd", data=dataTrain)


# Using Exploratory data aanalysis, few features can be dropped because they had no impact on the price prediction. Those features are removed with the function below.(Training set)




dataTrain.drop(['number_of_maintenance', 'duration_listed', 'engine_type','is_exchangeable'], axis = 1, inplace = True)


# Same features are removed for testing set since the data will be used to train the model




dataTrain.shape


# A descriptive analysis to check incorrect entries and anormalies. This is also used to give an overview of the numerical data. It is observed that most of the data has no incorrect entries.




dataTrain.describe()


# This is a check for categorical data, it is observed that all the data is within the range with no incorrect entries.




dataTrain['price_usd']





dataTrain.describe(include=['object'])


# ### Data Transformation
# Label encoding of categorical features in the training set. Label encoding is converting categorical data into numerical data since the model cant understand textual data.

# ----Data Preparation--------




from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
dataTrain.manufacturer_name = labelencoder.fit_transform(dataTrain.manufacturer_name)
dataTrain.transmission = labelencoder.fit_transform(dataTrain.transmission)
dataTrain.color = labelencoder.fit_transform(dataTrain.color)
dataTrain.engine_fuel = labelencoder.fit_transform(dataTrain.engine_fuel)
#dataTrain.engine_type = labelencoder.fit_transform(dataTrain.engine_type)
dataTrain.body_type = labelencoder.fit_transform(dataTrain.body_type)
dataTrain.has_warranty = labelencoder.fit_transform(dataTrain.has_warranty)
dataTrain.ownership = labelencoder.fit_transform(dataTrain.ownership)
dataTrain.type_of_drive = labelencoder.fit_transform(dataTrain.type_of_drive)
#dataTrain.is_exchangeable = labelencoder.fit_transform(dataTrain.is_exchangeable)





dataTrain.head(10)


# --Data Transfornation (normalization) ----
# z-score used for scaling down the features between the range of -1 and 1. This helps the model make better prediction as it is easy to understand. The scaling is applied to the training and testing set




# Calculate the z-score from with scipy
import scipy.stats as stats
dataTrain = stats.zscore(dataTrain)
dataTest = stats.zscore(dataTest)





dataTrain





dataTest


# Dividing the data for training and testing accordingly. X takes the all features while Y takes the target variable
# 
# We have 13 actual columns [0-12 index]; 12 are predictor variables and 1 is the target variable




x_train=dataTrain.iloc[:,0:11]
y_train=dataTrain.iloc[:,12]
x_test=dataTest.iloc[:,0:11]
y_test=dataTest.iloc[:,12]





x_train.head()





y_train.head()


# ## Fit Model
# ### Multiple Linear Regression
# Calling multiple linear regression model and fitting the training set




# importing train_test_split from sklearn
from sklearn.model_selection import train_test_split
# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)





#print the shape of train and test data after spltting
print (x_train.shape)
print (x_test.shape)





from sklearn.linear_model import LinearRegression

mlr = LinearRegression()
model_mlr = mlr.fit(x_train,y_train)





y_pred1 = model_mlr.predict(x_test)





#MLR Evaluation

MSE1 = mean_squared_error(y_test,y_pred1)
print('MSE is ', MSE1)


# ### Random Forest Regressor 
# Calling the random forest model and fitting the training data




rf = RandomForestRegressor()
modelrf=rf.fit(x_train,y_train)


# Prediction of car prices using the testing data




y_pred2 = rf.predict(x_test)





#RF Evaluation

MSE2 = mean_squared_error(y_test,y_pred2)
print('MSE is ', MSE2)





scores = [('MLR', MSE1),
          ('Random Forest', MSE2)
         ]         





MSE = pd.DataFrame(data = scores, columns=['Model', 'MSE Score'])
MSE





MSE.sort_values(by=(['MSE Score']), ascending=False, inplace=True)

f, axe = plt.subplots(1,1, figsize=(10,7))
sns.barplot(x = mae['Model'], y=MSE['MSE Score'], ax = axe)
axe.set_xlabel('Mean Squared Error', size=20)
axe.set_ylabel('Model', size=20)

plt.show()


# #Based on the MSE, it is concluded that the Random Forest is the best regression model for predicting the car price based on the 12 predictor variables 






