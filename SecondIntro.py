

import streamlit as st
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from PIL import Image

# #Set title

st.title('JL Data')

image=Image.open("C:\\Users\\Hp\\OneDrive\\Pictures\\logo-color.png")
st.image(image,use_column_width=True)

#Set subtitle

st.write("""# A simple Data App With Streamlit""")

st.write("""Let's Explore different classifiers and datasets""")

dataset_name=st.sidebar.selectbox('Select ddataset',('Breast Cancer','Iris','Wine'))

classifier_name=st.sidebar.selectbox('Select classifier',('SVM','KNN'))

def get_dataset(name):
	data=None
	if name=='Iris':
		data=datasets.load_iris()
	elif name=='Wine':
		data=datasets.load_wine()
	else:
		data=datasets.load_breast_cancer()
	x=data.data
	y=data.target

	return x,y

x,y=get_dataset(dataset_name)
st.dataframe(x)
st.write('Shape of your dataset is:',x.shape)
st.write('unique target variables:',len(np.unique(y)))	
		 

fig=plt.figure()
sns.boxplot(data=x,orient='h')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


plt.hist(x)
st.pyplot()


#BUILDING OUR ALGORITHM
def add_parameter(name_of_clf):
	params=dict()
	if name_of_clf=='SVM':
		c=st.sidebar.slider('C',0.01,15.0)
		params['C']=c
	else:
	    name_of_clf='KNN'
	    k=st.sidebar.slider('k',1,15)
	    params['k']=k
	return params

params=add_parameter(classifier_name)


#Accessing our classifier
def get_classifier(name_of_clf,params):
	clf=None
	if name_of_clf=='SVM':
		clf=SVC(C=params['C'])
	elif name_of_clf=='KNN':
		clf=KNeighborsClassifier(n_neighbors=params['k'])
	else:
		st.warning("you didn't select any option,please select at least one algo")
	return clf
	
			

clf=get_classifier(classifier_name,params)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
st.write(y_pred)

accuracy=accuracy_score(y_test,y_pred)
st.write('classifier_name:',classifier_name)
st.write('Accuracy for your model is:',accuracy)



	    	

