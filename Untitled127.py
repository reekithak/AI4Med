#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K


# In[7]:


base_model = DenseNet121();


# In[6]:


base_model.summary()# Print out the first five layers
layers_l = base_model.layers

print("First 5 layers")
layers_l[0:5]


# In[ ]:



print("First 5 layers")
layers_l[0:5]


# In[ ]:


# Print out the last five layers
print("Last 5 layers")
layers_l[-6:-1]


# In[ ]:


# Get the convolutional layers and print the first 5
conv2D_layers = [layer for layer in base_model.layers 
                if str(type(layer)).find('Conv2D') > -1]
print("The first five conv2D layers")
conv2D_layers[0:5]


# In[ ]:


print("The input has 3 channels")
base_model.input


# In[ ]:


# Print the number of output channels
print("The output has 1024 channels")
x = base_model.output
x


# In[ ]:


# Add a global spatial average pooling layer
x_pool = GlobalAveragePooling2D()(x)
x_pool


# In[ ]:




