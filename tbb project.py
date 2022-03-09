# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:40:57 2021

@author: tuncc
"""

import pandas as pd
import numpy as np
import  statistics
#1.1
strategy_games=pd.read_csv("strategy_games.csv")
print(strategy_games)
#1.2
pubg_mobile=strategy_games[strategy_games.loc[:,"Name"]=="PUBG MOBILE"]
print(pubg_mobile)
print(pubg_mobile.loc[:,["Average.User.Rating","User.Rating.Count"]])
#1.3            
names=strategy_games.Name[(strategy_games['Average.User.Rating']>=4,5)and (strategy_games["User.Rating.Count"] >=300000)]      
print(names) 
            
#1.4
strategy_games=pd.DataFrame(strategy_games)
FREE=strategy_games["Price"]==0
print(FREE)
strategy_games["FREE"]=FREE
print(strategy_games)

#1.5
import matplotlib.pyplot as plt                                                 
strategy_games.groupby(["FREE"])["Average.User.Rating"].mean().plot.bar(color=["red","blue"])
plt.ylim(3.95,4.1)
plt.title("Average User Rating Count Paid VS Free Games")
plt.show()

#1.6
paid=strategy_games[strategy_games.loc[:,"FREE"]==False]
print("Average.User.Rating is not free:",paid["Average.User.Rating"].mean())
free=strategy_games[strategy_games.loc[:,"FREE"]==True]
print("Average.User.Rating free:",free["Average.User.Rating"].mean())


#PART 2

import numpy as np
import pandas as pd
      
health_care=pd.read_csv("healthcare-dataset-stroke-data.csv",sep=";")
print(health_care)   
health_care=pd.DataFrame(health_care)

male=health_care[health_care.loc[:,"gender"]=="Male"]
male_stroke=male[male.loc[:,"stroke"]==1]
len(male_stroke)

female=health_care[health_care.loc[:,"gender"]=="Female"]
female_stroke=female[female.loc[:,"stroke"]==1]
len(female_stroke)

import seaborn as sns
sns.catplot(x="gender",hue="stroke",data=health_care,kind="count",palette="coolwarm")
plt.ylim(50,150);

married=health_care[health_care.loc[:,"ever_married"]=="Yes"]
married_stroke=married[married.loc[:,"stroke"]==1]
len(married_stroke)

single=health_care[health_care.loc[:,"ever_married"]=="No"]
single_stroke=single[single.loc[:,"stroke"]==1]
len(single_stroke)

import seaborn as sns
sns.catplot(x="ever_married",hue="stroke",data=health_care,kind="count",palette="pastel")
plt.ylim(10,300);


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
health=pd.read_csv("healthcare-dataset-stroke-data.csv")

#H0:Felç geçirenlerin glikoz seviyesi geçirmeyenlere göre daha düşüktür.(m1<=m2)H0:who have had a stroke have lower glucose levels than those who have not. (u1<=u2)
#H1:Felç geçirenlerin glikoz seviyesi geçirmeyenlere göre daha yüksektir(m1>m2)H1: who have had a stroke have higher glucose levels than those who have not.(u1>u2)
stroke=health[health["stroke"]==1]["avg_glucose_level"]
not_stroke=health[health["stroke"]==0]["avg_glucose_level"]
stroke_values=health[health["stroke"]==1]["avg_glucose_level"].values
not_stroke_values=health[health["stroke"]==0]["avg_glucose_level"].values
print(stats.ttest_ind(stroke_values,not_stroke_values,alternative="greater"))

import seaborn as sns
sns.catplot(x="stroke",y="avg_glucose_level",data=health,kind="bar")
plt.show()




   
    #PART 3 LINEAR REGRESSION EXAMPLE#
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
dataset = pd.read_csv('student_scores.csv')
dataset.head()
dataset.describe()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


plt.scatter(X_train , y_train, color='red')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')


plt.scatter(X_test , y_test,color="blue")
new_y_pred=regressor.predict(X_test)
plt.plot(X_test,new_y_pred,color='black')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')


print(regressor.intercept_)
print(regressor.coef_)


#COMMENT
y_pred = regressor.predict(X_test)
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
from sklearn import metrics
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
y_pred
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)





df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)



male=health[health.loc[:,"gender"]=="Male"]
male_stroke=male[male.loc[:,"stroke"]==1]
len(male_stroke)





#MULTIPLE REGRESSION














 
    
 
    
 
    