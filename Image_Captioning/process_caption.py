from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import json
import numpy as np 


#This function calculates the longest caption length
def calc_max_length(tensor):
    return max(len(t) for t in tensor)
   

# This function takes in all the captions, and does all the preprocesing steps
def proc_caption(train_captions):
    # Taking the top number of occuring words
    top_k = 10000

    # Creating an instance of Tensorflow tokenizer, which is primarily used for word processing
    # Eliminating all the special and unnecessary characters, and also making any word out of
    # the vocabulary as <unk>
    tokenizer = tf.keras.preprocessing.text.Tokenizer(  num_words=top_k,oov_token="<unk>",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    # this converts the words into numbers based on appearance
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    # creates an  word to index relationship Ex: 'the':21  'why':16 and 
    # index to word relationship Ex:  4:'yes,  5:'no'
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    # creating a dictionary of both these relationships
    word_index_dict = tokenizer.word_index
    index_word_dict = tokenizer.index_word
    # saving both the dictionaries, no need to compute every time
    np.save("D:\Final_year_project\Image_Captioning\wordindex.npy", word_index_dict)
    np.save("D:\Final_year_project\Image_Captioning\index_word.npy", index_word_dict)

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)

    return cap_vector, tokenizer, max_length