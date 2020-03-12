# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 22:42:53 2020

@author: kuchenov
"""
###################
#beutiful soup
from bs4 import BeautifulSoup as bsoup
import requests as rq
import re
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import seaborn as sns
import collections
import matplotlib as plt
import numpy as np
from  sklearn import model_selection
from  sklearn.preprocessing import minmax_scale
import random
import sklearn.metrics as rsqr


#############################
#####encode sequence
data = pd.read_csv('D:/Work/Computational/FPbase/20200228_FPbase_seq.csv') 
data.columns
data['Switch Type'].unique()

#clean data 
data['Seq_lenght'] = data.Sequence2.str.len()
data = data[data['Seq_lenght']!=5]
data = data[data['Seq_lenght']>150]
data = data[data['Ex max (nm)'].notna()]
data = data[data['Em max (nm)'].notna()]
data = data[data['Seq_lenght']<400]
data = data[data['Switch Type']!='pc']
data = data[data['Switch Type']!='ps']

#select rows
data = data[['Seq_lenght','Ex max (nm)','Em max (nm)','Sequence2']]


###the data does npot have non-fluorescent examples
#therefore, I have randomly shuffled the sequnces and assigne max Em and
#max Ex as zero (0) to train the model with non-fluorescent examples 
def randomize_sequence(data_df,column):
    lst = []
    for i in data_df[column]:
        i_list=[x for x in i]
        i_random = np.random.choice(i_list,len(i_list))
        lst.append(''.join(i_random))
    return lst
rendom_seq = randomize_sequence(data_df = data,column='Sequence2')
length = [len(x) for x in rendom_seq]
em_0 = [0]*len(rendom_seq)
ex_0 = [0]*len(rendom_seq)
df = pd.DataFrame({'Seq_lenght': length, 'Ex max (nm)': ex_0,'Em max (nm)': em_0, 'Sequence2':rendom_seq})

#Combine FPbase data with randomly shuffled data
data = pd.concat([data,df])

#Scale data between 0 and 1
data['Ex max (nm)'] = minmax_scale(data['Ex max (nm)'])
data['Em max (nm)'] = minmax_scale(data['Em max (nm)'])
#data['Stokes Shift (nm)'] = minmax_scale(data['Stokes Shift (nm)'])
#data['Quantum Yield'] = minmax_scale(data['Quantum Yield'])
#data['Brightness'] = minmax_scale(data['Brightness'])
#data['pKa'] = minmax_scale(data['pKa'])

#split train test data
x_train, x_test, y_train, y_test = model_selection.train_test_split(data['Sequence2'],
                                                                    data['Em max (nm)'],
                                                                    random_state=0)

#visualize dstribution
def histogram(data, bins=20):
    sns.set(style='whitegrid', palette="deep",
            font_scale=1.1, rc={"figure.figsize": [8, 5]})
    sns.distplot(data, norm_hist=False,
             kde=False, bins=bins, hist_kws=
             {"alpha": 1}).set(xlabel='lenght',
             ylabel='Count')

histogram(data['Seq_lenght'], bins=20)
###corelation scater plot plots
sns.pairplot(data, height=2.5)

#######visulize data
def get_code_freq(pandas_series):  
  codes = []
  for i in pandas_series: # concatination of all codes
      codes.extend(i)
  codes_dict= collections.Counter(codes)  
  df = pd.DataFrame({'Code': [*codes_dict], 'Freq':[codes_dict[t] for t in[*codes_dict]]})
  return df.sort_values('Freq', ascending=False).reset_index()[['Code', 'Freq']]

#code sequence
code_freq_train = get_code_freq(x_train)
code_freq_test = get_code_freq(x_test)

def plot_code_freq(df):
  #plt.title(f'Code frequency')
  sns.barplot(x='Code', y='Freq', data=df)

#final freq plot
plot_code_freq(code_freq_train)
plot_code_freq(code_freq_test)

#create nuber encoding
codes = code_freq_train.Code

def create_dict(codes):
  char_dict = {}
  for index, val in enumerate(codes):
    char_dict[val] = index+1
  return char_dict

dict_list =  create_dict(codes)

###transfer the sequence to the integer encoding
def integer_encoding(pandas_series):
  """
  - Encodes code sequence to integer values.
  - 22 amino acids are taken into consideration
    and rest 4 are categorized as 0.
  """
  encode_list = []
  for row in pandas_series.values:
    row_encode = []
    for code in row:
      row_encode.append(dict_list.get(code, 0))
    encode_list.append(np.array(row_encode))
  return encode_list
  
x_train_encode = integer_encoding(x_train)
x_test_encode = integer_encoding(x_test) 

#def randomize_sequence(array_list):
#    lst = []
#    for i in array_list:
#        i_random = np.random.choice(i,len(i))
#        lst.append(i_random)
#    return lst
#x_test_random = randomize_sequence(array_list=x_test_encode)


#encode short protins with 0 to bring to the same length
def fill_short_with_0(nparray,maxlength= data['Seq_lenght'].max()):
    fp = nparray[1]
    prot_list = []
    for fp in nparray:
        if len(fp) < (maxlength):
            long_prot = np.pad(fp, (0, maxlength-len(fp)), 'constant')
            prot_list.append(long_prot)
        else:
            prot_list.append(fp)
    return prot_list

###if the sequnce is shorter than max sequence in the data the lenght
    #is filled with 0 until the length is == max(length) in the data
x_train_encode_lng = fill_short_with_0(x_train_encode)
x_test_encode_lng = fill_short_with_0(x_test_encode)
#x_test_random_lng = fill_short_with_0(x_test_random)

x_train_ohe_3d = tf.keras.utils.to_categorical(x_train_encode_lng)
x_test_ohe_3d = tf.keras.utils.to_categorical(x_test_encode_lng)
#x_test_random_ohe_3d = tf.keras.utils.to_categorical(x_test_random_lng)

x_train_ohe =x_train_ohe_3d.reshape(len(x_train_ohe_3d),data['Seq_lenght'].max(), 22,1)
x_test_ohe =x_test_ohe_3d.reshape(len(x_test_ohe_3d),data['Seq_lenght'].max(),22,1)
#x_test_rendom_ohe =x_test_random_ohe_3d.reshape(len(x_test_random_ohe_3d),data['Seq_lenght'].max(),21,1)

#show random sequence:
def show_seq(array_seq=x_train_ohe_3d):
    fig = plt.pyplot.figure()
    random_index = np.random.randint(0, len(array_seq))
    ax = fig.add_subplot()
    ax.imshow(array_seq[random_index, :])
    plt.show()

show_seq(x_train_ohe_3d)


###build model for prediction
INIT_LR = 5e-3  # initial learning rate
BATCH_SIZE = 100
EPOCHS = 500

def build_model():
    model = models.Sequential([
            layers.Conv2D(8, (25, 22), activation='relu', name='Conv2D_1',
                          input_shape=(353, 22, 1)),
            layers.Dropout(0.25),                                       
            layers.AveragePooling2D((10, 1), name='AveragePooling2D_2'),
            #layers.Conv2D(8, (5, 1), activation='relu', name='Conv2D_3'),
            layers.Dropout(0.25),
            #layers.MaxPooling2D((10, 1),name='MaxPooling2D_4'),
            #layers.Conv2D(64, (3, 1), activation='relu', name='Conv2D_5'),
            layers.Flatten(),
            layers.Dense(4, activation='relu', name='Dense_6'),
            layers.Dropout(0.25),
            layers.Dense(1, activation='relu', name='Output_8')])
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


# clear default graph
model = build_model()
model.summary()


# scheduler of learning rate (decay with epochs)
def lr_scheduler(epoch):
    return INIT_LR * 0.99 ** epoch

##fit model
history = model.fit(x_train_ohe,
                    np.array(y_train).reshape(len(x_train_ohe)),
                    epochs = EPOCHS, validation_split = 0.15,
                    batch_size= BATCH_SIZE,
                    callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_scheduler)])


#Extration of training history
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#plot 
plt.pyplot.plot(hist['epoch'], hist['mse'])
plt.pyplot.plot(hist['epoch'], hist['val_mse'])


test_predictions = model.predict(x_test_ohe).flatten()
train_predictions = model.predict(x_train_ohe).flatten()

a = plt.pyplot.axes(aspect='equal')
plt.pyplot.scatter(y_test, test_predictions)
plt.pyplot.xlabel('True Values '+'[Em max (nm)]')
plt.pyplot.ylabel('Predictions [Em max (nm)]')

len(test_predictions)
len(y_test[y_test==0])
a = plt.pyplot.axes(aspect='equal')
plt.pyplot.scatter(y_train, train_predictions)
plt.pyplot.xlabel('True Values '+'[Em max (nm)]')
plt.pyplot.ylabel('Predictions [Em max (nm)]')

error = train_predictions - y_train
plt.pyplot.hist(error, bins = 40)
plt.pyplot.xlabel("Prediction Error [Ex max (nm)]")

error = test_predictions - y_test
plt.pyplot.hist(error, bins = 40)
plt.pyplot.xlabel("Prediction Error [Ex max (nm)]")
plt.pyplot.ylabel("Count")


##R2 calculation
rsqr.r2_score(y_train,train_predictions)
rsqr.r2_score(y_test,test_predictions)


