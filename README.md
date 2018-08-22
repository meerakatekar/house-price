# house-price
house price
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_excel("house sales.xlsx")
df.head()
df.info()
df["HOME.TYPE"].count()
df[df["HOME.TYPE"]=="Single Family Residential"].count()
df[df["HOME.TYPE"]=="Townhouse"].count()
df[df["HOME.TYPE"]=="Condo"].count()
df[df["HOME.TYPE"]=="Single Family Residential"]['LIST.PRICE'].max()
df[df["HOME.TYPE"]=="Townhouse"]['LIST.PRICE'].max()
df[df["HOME.TYPE"]=="Condo"]['LIST.PRICE'].max()
df[df['LIST.PRICE']==df["LIST.PRICE"]].min()
sns.set_style("whitegrid")
sns.countplot(x="HOME.TYPE",data=df)
sns.countplot(x="HOME.TYPE",hue="LIST.PRICE",data=df)
df['LIST.PRICE'].hist(color='green',bins=40,figsize=(8,4))
sns.boxplot(x="ZIP",y="HOME.TYPE",data=df,palette='winter')
hometype = pd.get_dummies(df['HOME.TYPE'],drop_first=True)
df.drop(['HOME.TYPE'],axis=1,inplace=True)
df = pd.concat([df,hometype],axis=1)
df.head()
from sklearn.model_selection import train_test_split
x = df[["SQFT","BEDS","BATHS","LOT.SIZE","YEAR.BUILT","DAYS.ON.MARKET","PARKING.TYPE","ZIP","Single Family Residential","Townhouse"]]
y = df["LIST.PRICE"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train,y_train)
pred = lm.predict(x_test)
plt.scatter(y_test,pred)
from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
