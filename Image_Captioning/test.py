from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
from batch_feature import batch_feature_processing, load_image, image_features_model
from process_caption import proc_caption     # dont use '-' for naming it doesnt work well
from model import CNN_Encoder, RNN_Decoder,Attention
from nltk.translate.bleu_score import sentence_bleu


# This function is used to load the image features
def map_func(img_name, cap):
  # adjust the path and the decode slicing with respect to how u have saved
  img_tensor = np.load('/home/cis/Documents/Image_captioning/vijay/Batch_Features'+img_name.decode('utf-8')[52:]+'.npy')
  return img_tensor, cap

# This function provides us with the necessary reference captoin for a particular image, which is needed in BLEU score evaluation
def ref_create(imageid):
  #annotation_file = os.path.abspath('.')+'/annotations/captions_train2014.json'
  annotation_file = 'annotations\captions_train2014.json'
 
  # Read the json file
  with open(annotation_file, 'r') as f:
      annotations = json.load(f)
  l =  []
  for annot in annotations['annotations']:
      # adding start and end tokens to each caption
    
      caption = '<start> ' + annot['caption'] + ' <end>'
      image_id = annot['image_id']
      if(image_id ==imageid):
        l.append(annot['caption'])
  return l

 
def main():

  #Training parameters
  total_size =100000
  top_k = 5000
  BATCH_SIZE = 64
  BUFFER_SIZE = 1000
  embedding_dim = 256
  units = 512
  vocab_size = top_k + 1 
  features_shape = 512
  attention_features_shape = 49
  
  
  # loading the training caption sequences and image name vectors
  train_captions,img_name_vector = np.load('traincaption_imgname.npy')
  # taking a subset of these and comverting to list
  img_name_vector = img_name_vector[:total_size].tolist()
  train_captions = train_captions[:total_size].tolist()
  #Enc_len = batch_feature_processing(img_name_vector)

  # creating an instance of CNN mdel
  image_features_extract_model = image_features_model()
  
  max_length = 51
  # processing the captions
  cap_vector, tokenizer, max_length = proc_caption(train_captions)

  # Create training and validation sets using an 80-20 split
  img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,cap_vector,
                                                                      test_size=0.2,random_state=0)  

  #MODEL STARTS HERE
  encoder = CNN_Encoder(embedding_dim)
  decoder = RNN_Decoder(embedding_dim, units, vocab_size)

  #OPTIMIZER AND LOSSS
  optimizer = tf.keras.optimizers.Adam()
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

  #CHECKPOINTS
  checkpoint_path = "\checkpoints\train100000"
  ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer = optimizer)
  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


  start_epoch = 0
  if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

  #EVALUATION
  def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    # resetting the hidden state of decoder
    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    # extract the image features
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    # passing the image features through an FC and ReLU layer
    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    # running loop for max length
    for i in range(max_length):
      # The decoder takes the image features, hidden state and initial input
      predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

      # The attention weights are stored every time step to show the change in attention
      attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

      # The predicted id are important as they are the keys to which the values are words
      # basically generate a number which corresponds to a number 
      predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
      print("predicted id:", predicted_id)
      # adding all the words together to make the caption
      result.append(tokenizer.index_word[predicted_id])

      # The loop ends when the model genrates the end token 
      if tokenizer.index_word[predicted_id] == '<end>':
          return result, attention_plot

      # reinitialising the decoder input
      dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot
  
  

  # captions on the validation set
  #ACTUAL EVALUATION
  print("len of image name val", len(img_name_val))
  # Taking a random number in the test dataset
  rid = np.random.randint(0, len(img_name_val))

  # taking the corresponding image from validation set
  image = img_name_val[rid]
  imageid = image[-10:-4]
  imageid = int(imageid)
  ref = ref_create(imageid)
  references = []
  for i in ref:
    l = i.split()
    references.append(l)
  
  real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])

  # testing from some random image
  #image = '/home/cis/Documents/Vijay/guy.jpeg'

  # BLEU score evaluation using NLTK library
  print("Reference Sentences:", references, "\n\n")
  print("Current real caption", real_caption, "\n\n")
  result, attention_plot = evaluate(image)
  print ('Prediction Caption:', ' '.join(result), "\n\n")
  print('Cumulative 1-gram BLEU-1: %f' % sentence_bleu(references, result, weights=(1, 0, 0, 0)))
  print('Cumulative 2-gram BLEU-2: %f' % sentence_bleu(references, result, weights=(0.5, 0.5, 0, 0)))
  print('Cumulative 3-gram BLEU-3: %f' % sentence_bleu(references,result, weights=(0.33, 0.33, 0.33, 0)))
  print('Cumulative 4-gram BLEU-4: %f' % sentence_bleu(references, result, weights=(0.25, 0.25, 0.25, 0.25)), "\n\n")
  
  
  # PLotting the attention weights along with words
  plot_attention(image, result, attention_plot)
  


#EVALUATION
def plot_attention(image, result, attention_plot):
  temp_image = np.array(Image.open(image))
  n=0
  # creating figure
  fig = plt.figure(num=n, figsize=(150,150))

  len_result = len(result)
  j =0
  for l in range(len_result):
    
    temp_att = np.resize(attention_plot[l], (8, 8))
    # max of 10 slots in a single figure
    if((l%10)==0 and l >0):
      n+=1
      j =0
      #creating subplots
      fig = plt.figure( num=n,figsize=(150,150))
      ax = fig.add_subplot(3,4, j+1)
    else:
      ax = fig.add_subplot(3,4,j+1)
      j+=1
    ax.set_title(result[l])
    img = ax.imshow(temp_image)
    # plotting the attention
    ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
    # the padding between the subtiles is important for presentation purposes
    plt.tight_layout(pad=50, w_pad=50.0, h_pad=50.0)
    

  plt.tight_layout(pad=50, w_pad=50.0, h_pad=50.0)
  plt.show()
  



if __name__== "__main__":
  # calling the main function
  main()





