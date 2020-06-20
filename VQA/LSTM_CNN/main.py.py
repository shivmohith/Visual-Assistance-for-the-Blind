"""
Note: Please run the code in google colab or jupyter notebook to avoid unnecessary errors
"""

"""
Note: Uncomment the below code only if run in google colab
Download the load2vec model
"""

# !python -m spacy download en_core_web_md

"""
Import the spacy library and load the word2vec model
"""

import spacy
nlp = spacy.load('en_core_web_md')
print ("Loaded WordVec")

"""
Note: The below code is applicable only if run in google colab
Mounting the drive to access the image features
"""
from google.colab import drive
drive.mount('/content/drive')
# %cd drive/My\ Drive/vqa
# !ls

"""
Importing the libraries
"""

import sys, warnings
warnings.filterwarnings("ignore")
from random import shuffle, sample
import pickle as pk
import gc
import operator
from collections import defaultdict
from itertools import zip_longest
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import LabelEncoder
import spacy

def freq_answers(training_questions, answer_train, images_train, upper_lim):
    """ Returns tuple of lists of filtered questions, answers, imageIDs based on the treshold limit.
    It filters out the samples based on the frequency of occurance of answer.

    Args:
        training_questions: training questions
        answers_train: corresponding training answers
        images_train: corresponding training imageIDs
        upper_lim: number of answers/classes
    """

    freq_ans = defaultdict(int)
    for ans in answer_train:
        freq_ans[ans] += 1

    sort_freq = sorted(freq_ans.items(), key=operator.itemgetter(1), reverse=True)[0:upper_lim]
    top_ans, top_freq = zip(*sort_freq)
    new_answers_train = list()
    new_questions_train = list()
    new_images_train = list()
    for ans, ques, img in zip(answer_train, training_questions, images_train):
        if ans in top_ans:
            new_answers_train.append(ans)
            new_questions_train.append(ques)
            new_images_train.append(img)

    return (new_questions_train, new_answers_train, new_images_train)

def grouped(iterable, n, fillvalue=None):
    """ Returns a zip object
    Groups the samples accorading to batch size.

    Args:
        iterable: samples to group
        n: batch size
        fillvalue: to fill empty values with
    """
    
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def get_answers_sum(answers, encoder):
    """ Returns tensorflow object
    Converts a class vector (integers) to binary class matrix

    Args:
        answers: answers in string literals
        encoder: a scikit-learn LabelEncoder object
    """
    
    assert not isinstance(answers, str)
    y = encoder.transform(answers)
    nb_classes = encoder.classes_.shape[0]
    Y = tf.keras.utils.to_categorical(y, nb_classes)
    return Y

"""
Loading the training image feature pickle file
"""

#pkl_file_val.close()
import pickle as pk
pkl_file = open('image_features.pkl', 'rb')
features= pk.load(pkl_file)

"""
Loading the testing image features pickle file
"""

#pkl_file.close()
pkl_file_val = open('image_features_val.pkl', 'rb')
features_val = pk.load(pkl_file_val)

def get_images_matrix(img_list):
	""" Returns a numpy array of size (nb_samples, nb_dimensions)
	Gets the 4096-dimensional CNN features for the given imageID

	Args:
			img_list: list of imageIDs
	"""

	image_matrix = np.zeros((len(img_list), 4096))
	index = 0
	for id in img_list:
		image_matrix[index] = features['%012d' %(int(id))]
		index = index + 1
	return image_matrix

def get_answers_matrix(answers, encoder):
	""" Returns numpy array of shape (nb_samples, nb_classes)
	Converts string objects to class labels

	Args:
			answers: asnwers in string format
			encoder: a scikit-learn LabelEncoder object
	"""

	assert not isinstance(answers, str)
	y = encoder.transform(answers) #string to numerical class
	nb_classes = encoder.classes_.shape[0]
	Y = np_utils.to_categorical(y, nb_classes)
	return Y

"""Returns a time series of word vectors for tokens in the question
	A numpy ndarray of shape: (nb_samples, timesteps, word_vec_dim)"""

def get_questions_tensor_timeseries(questions, nlp, timesteps):
	""" Returns numpy ndarray of shape: (nb_samples, timesteps, word_vec_dim)
	Returns a time series of word vectors for tokens in the question

	Args:
			questions: questions
			nlp: word2vec model
			timesteps: uniform length of question
	"""
	
	assert not isinstance(questions, str)
	nb_samples = len(questions)
	word_vec_dim = nlp(questions[0])[0].vector.shape[0]
	questions_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
	for i in range(len(questions)):
		tokens = nlp(questions[i])
		for j in range(len(tokens)):
			if j<timesteps:
				questions_tensor[i,j,:] = tokens[j].vector
	return questions_tensor

"""
Load the dataset and related files
"""

training_questions     = open("preprocessed/v1/ques_train.txt","rb").read().decode('utf8').splitlines()
training_questions_len = open("preprocessed/v1/ques_train_len.txt","rb").read().decode('utf8').splitlines()
answers_train          = open("preprocessed/v1/answer_train.txt","rb").read().decode('utf8').splitlines()
images_train           = open("preprocessed/v1/images_coco_id.txt","rb").read().decode('utf8').splitlines()
img_ids                = open('preprocessed/v1/coco_vgg_IDMap.txt').read().splitlines()

sample(list(zip(images_train, training_questions, answers_train)), 5)

"""
Garbage collection
"""

gc.collect()

""" 
Filtering the training dataset based on the number of classes and its frequency of occurance
"""

upper_lim = 1000 
print (len(training_questions), len(answers_train),len(images_train))
training_questions, answers_train, images_train = freq_answers(training_questions, 
                                                               answers_train, images_train, upper_lim)
training_questions_len, training_questions, answers_train, images_train = (list(t) for t in zip(*sorted(zip(training_questions_len, 
                                                                                                          training_questions, answers_train, 
                                                                                                          images_train))))
print (len(training_questions), len(answers_train),len(images_train))

"""
Gets the unique answers and store them in a .sav file
"""

lbl = LabelEncoder()
lbl.fit(answers_train)
nb_classes = len(list(lbl.classes_))
print(nb_classes)
pk.dump(lbl, open('preprocessed/v1/label_encoder_lstm.sav','wb'))

""" 
Training parameters and model configurations
"""

batch_size               =      256
img_dim                  =     4096
word2vec_dim             =      300
#max_len                 =       30 # Required only when using Fixed-Length Padding

num_hidden_nodes_mlp     =     1024
num_hidden_nodes_lstm    =      512
num_layers_mlp           =        3
num_layers_lstm          =        3
dropout                  =       0.5
activation_mlp           =     'tanh'

num_epochs               =         150 #number of epochs

"""
Building the image model
"""

image_model = tf.keras.Sequential([tf.keras.layers.Reshape(input_shape = (img_dim,), target_shape=(img_dim,), name='Feeding_image_vectors_size_4096')], name='Image_Model')
image_model.add(tf.keras.layers.Dense(num_hidden_nodes_mlp, kernel_initializer='uniform', name = 'Image_MLP_Hidden_layer_size_1024'))
image_model.add(tf.keras.layers.Activation('tanh', name='Image_MLP_Activation_tanh'))
image_model.add(tf.keras.layers.Dropout(0.5, name='Image_MLP_Dropout_0.5'))
image_model.summary()

""" 
Building the question model
"""

language_model = tf.keras.Sequential([tf.keras.layers.LSTM(num_hidden_nodes_lstm,return_sequences=True, input_shape=(None, word2vec_dim), name='Feeding_question_vectors_to_LSTM_Layer_1'),
                                      tf.keras.layers.LSTM(num_hidden_nodes_lstm, return_sequences=True, name='LSTM_layer_2'),
                                      tf.keras.layers.LSTM(num_hidden_nodes_lstm, return_sequences=False, name='LSTM_layer_3')
                                      ], name = 'Language_Model')
language_model.add(tf.keras.layers.Dense(num_hidden_nodes_mlp, kernel_initializer='uniform', name = 'Question_MLP_Hidden_layer_size_1024'))
language_model.add(tf.keras.layers.Activation('tanh', name='Question_MLP_Activation_tanh'))
language_model.add(tf.keras.layers.Dropout(0.5, name='Question_MLP_Dropout_0.5'))

# for i in range(num_layers_lstm-2):
#     language_model.add()
# language_model.add(LSTM(num_hidden_nodes_lstm, return_sequences=False))

language_model.summary()

"""
Concatenating image model and question model and building the final model
"""

upper_lim = 1000 #  

merged=tf.keras.layers.concatenate([language_model.output,image_model.output], axis =-1, name='Merging_language_model_and_image_model')

model =tf.keras.Sequential(name='CNN_LSTM_Model')(merged)
model = tf.keras.layers.Dense(num_hidden_nodes_mlp, kernel_initializer='uniform', name = 'Combined_MLP_Hidden_layer_size_1024')(model)
model = tf.keras.layers.Activation('tanh', name='Combined_MLP_Activation_tanh')(model)
model = tf.keras.layers.Dropout(0.5, name='Combined_MLP_Dropout_0.5')(model)
model = tf.keras.layers.Dense(upper_lim, name='Fully_Connected_Output_layer_size_1000')(model)
out =   tf.keras.layers.Activation("softmax", name='Softmax_Output_Probablities')(model)
model = tf.keras.Model([language_model.input, image_model.input], out, name='LSTM_CNN_Model')

""" 
Compile the model
"""

#model.load_weights('weights_adam/LSTM_1000classes_epoch_21.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='rms_prop')
#tf.keras.utils.plot_model(model, to_file='CNN_LSTM_model_v2.png')

loss_epoch = [] #to store the loss per epoch

"""
Training starts here
"""

for k in range(num_epochs):
    loss_per_epoch = 0
    progbar = tf.keras.utils.Progbar(len(training_questions))
    for ques_batch, ans_batch, im_batch in zip(grouped(training_questions, batch_size, 
                                                       fillvalue=training_questions[-1]), 
                                               grouped(answers_train, batch_size, 
                                                       fillvalue=answers_train[-1]), 
                                               grouped(images_train, batch_size, fillvalue=images_train[-1])):
        timestep = len(nlp(ques_batch[-1]))
        X_ques_batch = get_questions_tensor_timeseries(ques_batch, nlp, timestep)
        # print (X_ques_batch.shape)
        X_ques_batch = np.array(X_ques_batch)
        # X_img_batch = np.zeros((batch_size,4096))
        #print(im_batch)
        X_img_batch = get_images_matrix(im_batch)
        # print(X_img_batch) 
        X_img_batch = np.array(X_img_batch) 
        Y_batch = get_answers_sum(ans_batch, lbl)
        Y_batch = np.array(Y_batch)
        loss = model.train_on_batch([X_ques_batch, X_img_batch], Y_batch)

        loss_per_epoch = loss_per_epoch + loss
     
        progbar.add(batch_size, values=[('train loss', loss), ('epoch', k)])
        
    if k%1 == 0:
        model.save_weights("weights_adam/LSTM_1000classes" + "_epoch_{:02d}.hdf5".format(k))

    loss_epoch.append(loss_per_epoch)
    np.save('weights_adam/loss_22.npy', np.array(loss_epoch))

model.save_weights("weights_adam/LSTM_1000classes" + "_epoch_{:02d}.hdf5".format(k))

"""
Loading the weights and compiling the model to test
"""

model.load_weights('weights_plot/LSTM_1000classes_epoch_70.hdf5') 
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print ("Model Loaded with Weights") 
# model.summary()

"""
Loading the testing dataset and related files
"""

val_imgs = open('preprocessed/v1/val_images_coco_id.txt','rb').read().decode('utf-8').splitlines()
val_ques = open('preprocessed/v1/ques_val.txt','rb').read().decode('utf-8').splitlines()
val_ans  = open('preprocessed/v1/answer_val.txt','rb').read().decode('utf-8').splitlines()
print (len(val_ques), len(val_imgs),len(val_ans))

"""
Filtering the testing dataset based on the number of answers and its frequency of occurance
"""

upper_lim = 1000 
val_ques, val_ans, val_imgs = freq_answers(val_ques, val_ans, val_imgs, upper_lim)
print (len(val_ques), len(val_imgs),len(val_ans))

"""
Loading the label encoder for classes
"""

label_encoder = pk.load(open('preprocessed/v1/label_encoder_lstm.sav','rb'))

"""
Testing starts here
"""

y_pred = []
batch_size = 256 
progbar = tf.keras.utils.Progbar(len(val_ques))
for qu_batch,an_batch,im_batch in zip(grouped(val_ques, batch_size, 
                                                   fillvalue=val_ques[0]), 
                                           grouped(val_ans, batch_size, 
                                                   fillvalue=val_ans[0]), 
                                           grouped(val_imgs, batch_size, 
                                                   fillvalue=val_imgs[0])):
    timesteps = len(nlp(qu_batch[-1]))
    X_ques_batch = get_questions_tensor_timeseries(qu_batch, nlp, timesteps)
    # X_i_batch = np.zeros((batch_size,4096))
    X_i_batch = get_images_matrix(im_batch)
    X_batch = [X_ques_batch, X_i_batch]
    y_predict = model.predict(X_batch, verbose=0)
    y_predict = np.argmax(y_predict,axis=1)
    y_pred.extend(label_encoder.inverse_transform(y_predict))
    progbar.add(batch_size)

"""
Calculating the accuracy and saving the results in a text file
"""

correct_val = 0.0
total = 0 

correct_total = 0.0
total = 0.0

correct_yes_or_no = 0.0
total_yes_or_no = 0.0

correct_number = 0.0
total_number = 0.0

correct_other = 0.0
total_other = 0.0

f1 = open('proper_results_v2/res_v2_lstm_cnn_1000classes_70epoch.txt','w') 

for pred, truth, ques, img in zip(y_pred, val_ans, val_ques, val_imgs):
    t_count = 0
    for _truth in truth.split(';'):
        if pred == truth:
            t_count += 1 
    if t_count >=1:
        correct_val +=1
    else:
        correct_val += float(t_count)/3

    total +=1

    if truth == 'yes' or truth == 'no':
      if pred == truth:
        correct_yes_or_no += 1
      total_yes_or_no += 1
    
    if truth.isnumeric():
      if pred == truth:
        correct_number += 1
      total_number += 1
    
    else:
      if pred == truth:
        correct_other += 1
      total_other += 1

    try:
        f1.write(str(ques))
        f1.write('\n')
        f1.write(str(img))
        f1.write('\n')
        f1.write(str(pred))
        f1.write('\n')
        f1.write(str(truth))
        f1.write('\n')
        f1.write('\n')
    except:
        pass

print('Final Accuracy is ' + str(correct_val/total))
print('Yes or no Accuracy is ' + str(correct_yes_or_no/total_yes_or_no))
print('Number Accuracy is ' + str(correct_number/total_number))
print('Other Accuracy is ' + str(correct_other/total_other))

f1.write('Final Accuracy is ' + str(correct_val/total))
f1.write('\n')
f1.write('Yes or no Accuracy is ' + str(correct_yes_or_no/total_yes_or_no))
f1.write('\n')
f1.write('Number Accuracy is ' + str(correct_number/total_number))
f1.write('\n')
f1.write('Other Accuracy is ' + str(correct_other/total_other))
f1.write('\n')

f1.close()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

"""
Loading the saved losses
"""

loss1 = np.load('weights_plot/loss_1.npy').tolist()
loss18 = np.load('weights_plot/loss_18.npy').tolist()[1:]
loss36 = np.load('weights_plot/loss_36.npy').tolist()[1:]
loss54 = np.load('weights_plot/loss_54.npy').tolist()[1:]
# loss78 = np.load('weights_plot/loss_78.npy').tolist()

loss = loss1 + loss18 + loss36 + loss54 

print(loss)
print(len(loss))

"""
Plotting the loss vs epoch graph
"""

e = np.arange(1, len(loss)+1, 1)
print(e)

plt.plot(e, loss)
plt.xlabel('epoch')
plt.ylabel('loss per epoch')
plt.title('loss vs epoch')

patch4 = mpatches.Patch(color = 'white',label='Model: CNN + LSTM')
patch = mpatches.Patch(color = 'white',label='learning rate: 0.001')
patch2 = mpatches.Patch(color = 'white',label='optimizer: RMS Prop') 
patch3 = mpatches.Patch(color = 'white',label='No. of training samples: 215407')

plt.legend(handles=[patch4,patch, patch2,patch3])
plt.show()
plt.savefig('proper_results_v2/CNN_LSTM_LOSS')