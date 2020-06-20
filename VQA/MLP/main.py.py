"""
Note: Please run the code in google colab or jupyter notebook to avoid unnecessary errors
"""

"""
Note: Uncomment the below code only if run in google colab
Download the load2vec model
"""

#!python -m spacy download en_core_web_md

"""
Import the spacy library and load the word2vec model
"""

import spacy
nlp = spacy.load('en_core_web_md')

"""
Note: The below code is applicable only if run in google colab
Mounting the drive to access the image features
"""

from google.colab import drive
drive.mount('/content/drive')
# %cd drive/My\ Drive/vqa
#!ls

"""
Importing the libraries
"""

import sys, warnings
import tensorflow as tf
import keras
warnings.filterwarnings("ignore")
from random import shuffle, sample
import pickle as pk
import gc
import operator
import numpy as np
import pandas as pd
import scipy.io
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from progressbar import Bar, ETA, Percentage, ProgressBar    
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from itertools import zip_longest

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
pkl_file = open('image_features.pkl', 'rb')
features= pk.load(pkl_file)

"""
Loading the testing image features pickle file
"""

# pkl_file.close()
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

def get_questions_sum(questions, nlp):
    """ Returns numpy array (nb_samples, word_2_vec_dim)
    Computes the question features using the word2vec model

    Args:
        questions: questions
        nlp: word2vec model
    """

    assert not isinstance(questions, str)
    nb_samples = len(questions)
    word2vec_dim = nlp(questions[0])[0].vector.shape[0]
    ques_matrix = np.zeros((nb_samples, word2vec_dim))
    for index in range(len(questions)):
        tokens = nlp(questions[index])
        for j in range(len(tokens)):
            ques_matrix[index,:] += tokens[j].vector

    return ques_matrix

"""
Load the dataset and related files
"""

training_questions = open("v1/ques_train.txt","rb").read().decode('utf8').splitlines()
answers_train      = open("v1/answer_train.txt","rb").read().decode('utf8').splitlines()
images_train       = open("v1/images_coco_id.txt","rb").read().decode('utf8').splitlines()
img_ids            = open('v1/coco_vgg_IDMap.txt').read().splitlines()
vgg_path           = "/floyd/input/vqa_data/coco/vgg_feats.mat"
print (len(training_questions), len(answers_train),len(images_train))

sample(list(zip(images_train, training_questions, answers_train)), 5)

"""
Garbage collection
"""

gc.collect()

""" 
Filter the dataset based on the number of classes and its frequency of occurance
"""

upper_lim = 1000 #Number of most frequently occurring answers in COCOVQA (Coverting >85% of the total data)

training_questions, answers_train, images_train = freq_answers(training_questions, answers_train, images_train, upper_lim)
print (len(training_questions), len(answers_train),len(images_train))

""" 
Training parameters and model configurations
"""

num_hidden_units  = 1000
num_hidden_layers = 2
batch_size        = 256
dropout           = 0.5
activation        = 'tanh'
img_dim           = 4096
word2vec_dim      = 300

num_epochs =   100 # number of epochs

"""
Gets the unique answers and store them in a .sav file
"""

lbl = LabelEncoder()
lbl.fit(answers_train)
nb_classes = len(list(lbl.classes_))
pk.dump(lbl, open('v1/label_encoder_mlp.sav','wb'))

"""
Building the MLP VQA model and compiling the model
"""

model = Sequential(name = "MLP model")
model.add(Dense(num_hidden_units, input_dim=word2vec_dim+img_dim, kernel_initializer='uniform',name="feeding_comined_image_question_vector"))
model.add(Dropout(dropout,name="Dropout_1_0.5"))
for i in range(num_hidden_layers):
    name_d = "MLP_"+str(i+1)+"_Hidden_layer_size_1000"
    model.add(Dense(num_hidden_units, kernel_initializer='uniform',name = name_d))
    name_a = "Activation_"+str(i+1)+"_tanh"
    model.add(Activation(activation,name = name_a))
    temp = "Dropout_"+str(i+2)+"_0.5"
    model.add(Dropout(dropout,name=temp))
model.add(Dense(nb_classes, kernel_initializer='uniform',name = "MLP_output_layer_size_1000"))
model.add(Activation('softmax',name = "softmax_output_Probabilities"))
model.load_weights('weights_plot/MLP_1000classes_epoch_62.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#tensorboard = TensorBoard(log_dir='/output/Graph', histogram_freq=0, write_graph=True, write_images=True)
#model.summary()

tf.keras.utils.plot_model(model, to_file = 'mlp_model.png')

loss_epoch = [] #to store the loss per epoch

"""
Training starts here
"""

for k in range(num_epochs):
    index_shuffle = list(range(len(training_questions)))
    loss_per_epoch = 0
    shuffle(index_shuffle)
    training_questions = [training_questions[i] for i in index_shuffle]
    answers_train = [answers_train[i] for i in index_shuffle]
    images_train = [images_train[i] for i in index_shuffle]
    progbar = generic_utils.Progbar(len(training_questions))
    for ques_batch, ans_batch, im_batch in zip(grouped(training_questions, batch_size, 
                                                       fillvalue=training_questions[-1]), 
                                               grouped(answers_train, batch_size, 
                                                       fillvalue=answers_train[-1]), 
                                               grouped(images_train, batch_size, fillvalue=images_train[-1])):
        X_ques_batch = get_questions_sum(ques_batch, nlp)
        X_img_batch = get_images_matrix(im_batch)
        X_batch = np.hstack((X_ques_batch, X_img_batch))
        Y_batch = get_answers_sum(ans_batch, lbl)
        #loss = model.train_on_batch(X_batch, Y_batch,callbacks= [tensorboard])
        loss = model.train_on_batch(X_batch, Y_batch)

        loss_per_epoch = loss_per_epoch + loss
        progbar.add(batch_size, values=[('train loss', loss), ('Epoch', k)])
    if k%1 == 0:
        model.save_weights("weights_plot/MLP_1000classes" + "_epoch_{:02d}.hdf5".format(k))

    loss_epoch.append(loss_per_epoch)
    np.save('weights_plot/loss_5.npy', np.array(loss_epoch))

model.save_weights("weights_plot/MLP_1000classes"+"_epoch_{:02d}.hdf5".format(k))

"""
Loading the weights and compiling the model to test
"""

model.load_weights('weights_mlp_v2/MLP_epoch_110.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

"""
Loading the testing dataset and related files
"""

val_imgs = open('v1/val_images_coco_id.txt','rb').read().decode('utf-8').splitlines()
val_ques = open('v1/ques_val.txt','rb').read().decode('utf-8').splitlines()
val_ans  = open('v1/answer_val.txt','rb').read().decode('utf-8').splitlines()

"""
Loading the label encoder for classes
"""

label_encoder = pk.load(open('v1/label_encoder_mlp.sav','rb'))

"""
Filtering the testing dataset based on the number of answers and its frequency of occurance
"""

upper_lim = 1000 
val_ques, val_ans, val_imgs = freq_answers(val_ques, val_ans, val_imgs, upper_lim)
print (len(val_ques), len(val_imgs),len(val_ans))

"""
Testing starts here
"""

y_pred = []
batch_size = 256 
progbar = tf.keras.utils.Progbar(len(val_ques))
for qu_batch,an_batch,im_batch in zip(grouped(val_ques, batch_size, fillvalue=val_ques[0]), grouped(val_ans, batch_size, fillvalue=val_ans[0]), grouped(val_imgs, batch_size, fillvalue=val_imgs[0])):
    X_q_batch = get_questions_matrix(qu_batch, nlp)
    X_i_batch = get_images_matrix2(im_batch)
    X_batch = np.hstack((X_q_batch, X_i_batch))
    y_predict = model.predict_classes(X_batch, verbose=0)
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

f1 = open('proper_results/res_mlp_1000classes_100epoch.txt','w') 

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