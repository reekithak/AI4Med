#!/usr/bin/env python
# coding: utf-8

# # Chest X-Ray Medical Diagnosis with Deep Learning - LUNG DISEASES
# 

# In[ ]:





# In[ ]:





# # Submitted by Akhil Sanker - RA1811026020035  , Surya k - RA1811026020006 , Melvin Abraham - RA1811026020029

# # CSE - AI/ML [2nd year]

# In[22]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

from keras.models import load_model

import util


# #### Read in the data
# Let's open these files using the [pandas](https://pandas.pydata.org/) library

# In[23]:


train_df = pd.read_csv("nih/train-small.csv")
valid_df = pd.read_csv("nih/valid-small.csv")

test_df = pd.read_csv("nih/test.csv")

train_df.head()


# In[24]:


labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']


# In[25]:


# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def check_for_leakage(df1, df2, patient_col):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs
    
    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """
    x=patient_col
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    df1_patients_unique = df1[patient_col].values
    df2_patients_unique = df2[patient_col].values
    df1_set=set(df1_patients_unique)
    df2_set=set(df2_patients_unique)
    patients_in_both_groups = list(df1_set.intersection(df2_set))

    # leakage contains true if there is patient overlap, otherwise false.
    if len(patients_in_both_groups)>0:
        leakage= True
    else:
        leakage = False# boolean (true if there is at least 1 patient in both groups)
    
    ### END CODE HERE ###
    
    return leakage


# Run the next cell to check if there are patients in both train and test or in both valid and test.

# In[26]:


print("leakage between train and test: {}".format(check_for_leakage(train_df, test_df, 'PatientId')))
print("leakage between valid and test: {}".format(check_for_leakage(valid_df, test_df, 'PatientId')))


# <a name='2-2'></a>
# ### Preparing Images

# In[27]:


def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """        
    print("getting train generator...") 
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator


# #### Build a separate generator for valid and test sets
# 
# Now we need to build a new generator for validation and testing data. 
# 
# **Why can't we use the same generator as for the training data?**
# 
# Look back at the generator we wrote for the training data. 
# - It normalizes each image **per batch**, meaning that it uses batch statistics. 
# - We should not do this with the test and validation data, since in a real life scenario we don't process incoming images a batch at a time (we process one image at a time). 
# - Knowing the average per batch of test data would effectively give our model an advantage.  
#     - The model should not have any information about the test data.
# 
# What we need to do is normalize incoming test data using the statistics **computed from the training set**. 
# * We implement this in the function below. 
# * There is one technical note. Ideally, we would want to compute our sample mean and standard deviation using the entire training set. 
# * However, since this is extremely large, that would be very time consuming. 
# * In the interest of time, we'll take a random sample of the dataset and calcualte the sample mean and sample standard deviation.

# In[28]:


def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for validation set and test test set using 
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=IMAGE_DIR, 
        x_col="Image", 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator


# In[29]:


IMAGE_DIR = "nih/images-small/"
train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", labels)
valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels)


# Let's peek into what the generator gives our model during training and validation. We can do this by calling the `__get_item__(index)` function:

# In[30]:


x, y = train_generator.__getitem__(0)
plt.imshow(x[0]);


# <a name='3-1'></a>
# ### Class Imbalance
# One of the challenges with working with medical diagnostic datasets is the large class imbalance present in such datasets. Let's plot the frequency of each of the labels in our dataset:

# In[31]:


plt.xticks(rotation=90)
plt.bar(x=labels, height=np.mean(train_generator.labels, axis=0))
plt.title("Frequency of Each Class")
plt.show()


# <a name='Ex-2'></a>
# ### Computing Class Frequencies
# 

# In[32]:


labels


# <details>    
# <summary>
#     <font size="3" color="darkgreen"><b>Hints</b></font>
# </summary>
# <p>
# <ul>
#     <li> Use numpy.sum(a, axis=), and choose the axis (0 or 1) </li>
# </ul>
# </p>
# 

# In[33]:


# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # total number of patients (rows)
    N = len(labels)
    
    positive_frequencies = np.sum(labels,axis=0)/N
    negative_frequencies = 1-positive_frequencies

    ### END CODE HERE ###
    return positive_frequencies, negative_frequencies


# In[34]:


freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
freq_pos


# Let's visualize these two contribution ratios next to each other for each of the pathologies:

# In[35]:


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
plt.xticks(rotation=90)
f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)


# In[36]:


pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights


# Let's verify this by graphing the two contributions next to each other again:

# In[37]:


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} 
                        for l,v in enumerate(neg_contribution)], ignore_index=True)
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label" ,data=data);


# <a name='Ex-3'></a>
# ###  Weighted Loss
# Fill out the `weighted_loss` function below to return a loss function that calculates the weighted loss for each batch.
# 

# In[38]:


import tensorflow as tf


# In[39]:


# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Tensor): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

        for i in range(len(pos_weights)):
            
            # for each class, add average weighted loss for that class 
            pos_avg = -1*(pos_weights[i]*y_true[:,i]*tf.keras.backend.log(y_pred[:,i]+epsilon))
            neg_avg = -1*(neg_weights[i]*(1-y_true[:,i])*tf.keras.backend.log(1-y_pred[:,i]+epsilon))
            loss =loss + tf.keras.backend.mean(pos_avg + neg_avg) 
             #complete this line
        return loss
    
        ### END CODE HERE ###
    return weighted_loss


# <a name='3-3'></a>
# ### DenseNet121
# 
# We will use a pre-trained [DenseNet121](https://www.kaggle.com/pytorch/densenet121) model which we can load directly from Keras and then add two layers on top of it:
# 1. A `GlobalAveragePooling2D` layer to get the average of the last convolution layers from DenseNet121.
# 2. A `Dense` layer with `sigmoid` activation to get the prediction logits for each of our classes.
# 

# In[40]:


# create the base pre-trained model
base_model = DenseNet121(weights='./nih/densenet.hdf5', include_top=False)

x = base_model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)

# and a logistic layer
predictions = Dense(len(labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))


# <a name='4'></a>
# ## 4 Training
# 
# 
# 

# In[ ]:


history = model.fit_generator(train_generator, 
                              validation_data=valid_generator,
                              steps_per_epoch=100, 
                              validation_steps=25, 
                              epochs = 3)


# In[42]:


model.load_weights("./nih/pretrained_model.h5")


# <a name='5'></a>
# ## 5 Prediction and Evaluation

# In[43]:


predicted_vals = model.predict_generator(test_generator, steps = len(test_generator))


# <a name='5-1'></a>
# ### 5.1 ROC Curve and AUROC
# 

# In[44]:


auc_rocs = util.get_roc_curve(labels, predicted_vals, test_generator)


# <a name='5-2'></a>
# ### 5.2 Visualizing Learning with GradCAM 
# 

# First we will load the small training set and setup to look at the 4 classes with the highest performing AUC measures.

# In[45]:


df = pd.read_csv("nih/train-small.csv")
IMAGE_DIR = "nih/images-small/"

# only show the lables with top 4 AUC
labels_to_show = np.take(labels, np.argsort(auc_rocs)[::-1])[:4]


# Now let's look at a few specific images.

# In[46]:


util.compute_gradcam(model, '00008270_015.png', IMAGE_DIR, df, labels, labels_to_show)


# In[47]:


util.compute_gradcam(model, '00011355_002.png', IMAGE_DIR, df, labels, labels_to_show)


# In[48]:


util.compute_gradcam(model, '00029855_001.png', IMAGE_DIR, df, labels, labels_to_show)


# In[49]:


util.compute_gradcam(model, '00005410_000.png', IMAGE_DIR, df, labels, labels_to_show)


# In[ ]:


base_model.save("model.h5")


# 
# 

# In[ ]:





# In[ ]:




