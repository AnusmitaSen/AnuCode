#!/usr/bin/env python
# coding: utf-8

# Decision Tree Classifier for given iris dataset-ANUSMITA SEN
# 
# The iris samples in question fall into three species: Iris setosa , Iris versicolor and Iris Virginica .
# 
# Decision tree models are the simplest form of tree-based models, and are arguably the simplest form of supervised multivariate classification models . A series of logical tests (generally in the form of boolean comparisions) are applied to the sample entries and their resulting subsets in turn to arrive at a final decision . It is very easy to visualise the decision process in a simple flowchart to trace the rational of every assignment made by a decision model , making it among the most interpretable models .

# In[2]:


# Importing libraries in Python
import sklearn.datasets as datasets
import pandas as pd

# Loading the iris dataset
iris=datasets.load_iris()

# Forming the iris dataframe
df=pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head(5))

y=iris.target
print(y)


# Now let us define the Decision Tree Algorithm

# In[ ]:


# Defining the decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(df,y)


# Let us visualize the Decision Tree to understand it better

# In[ ]:


# Install required libraries
import pydotplus
import graphviz
import sklearn


# In[ ]:


# Import necessary libraries for graph viz
get_ipython().system('pip install six')
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


# In[ ]:


# Visualize the graph

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, feature_names=iris.feature_names,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# ![image.png](attachment:image.png)

# In[ ]:




