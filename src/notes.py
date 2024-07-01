# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                     notes
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# scatter chart
# col1=Xy_train_DF['trestbps']
# col2=Xy_train_DF['chol']
# labelx='label x'
# labely='label y'
# plt.scatter(col1,col2)
# plt.xlabel(labelx)
# plt.ylabel(labely)
 
# bar chart
# high1=1
# high2=2
# high3=3
# col1="A"
# col2="B"
# col3="C"
# labelx='label x'
# labely='label y'
# plt.bar(x = [col1,col2,col3], height=[high1, high2, high3])
# plt.xlabel(labelx)
# plt.ylabel(labely)
 
# box-plot chart
# sns.boxplot(x = 'y', y='age', data=Xy_train_DF)
  
# correlation matrix
# dataFrame = Xy_train_DF.loc[:,['trestbps','chol','thalach','oldpeak']]
# sns.heatmap(dataFrame.corr(), annot=True, cmap='coolwarm')
# plt.show()

# useful notes
# Xy_train_DF.loc[Xy_train_DF['age'] > 80 , 'age']= 80    #fixing age problem
# trainDF.info()                                          #print info
# trainDF.describe()                                      #print describe
# plt.show()                                              #print plot
# print (Xy_train_DF.head())                              #print first 5 rows
# print (Xy_train_DF.tail())                              #print last 5 rows
# Xy_train_DF = Xy_train_DF.loc[:,['slope','cp','thalach']]
# print(Xtrain_DF.shape[0])
