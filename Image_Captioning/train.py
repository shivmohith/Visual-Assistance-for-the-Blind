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


### This function is used to load the image feature npy files, it is used in the tensorflow data pipeline ###
def map_func(img_name, cap):
  img_tensor = np.load('/home/cis/Documents/Vijay/Batch_Features'+img_name.decode('utf-8')[35:]+'.npy')
  return img_tensor, cap

 
### This function creates the image name vector list along
#    with the  corresponding captions from the json file, shuffles them 
#    and takes a subset of them                                        ###
def preprocess(total_size):
  annotation_file = os.path.abspath('.')+'/annotations/captions_train2014.json'
  PATH = os.path.abspath('.')+'/train2014/'
  # Read the json file
  with open(annotation_file, 'r') as f:
      annotations = json.load(f)

  all_img_name_vector = []
  all_captions = []

  for annot in annotations['annotations']:
      # adding start and end tokens to each caption
    
      caption = '<start> ' + annot['caption'] + ' <end>'
      image_id = annot['image_id']
      # creating a full image path
      full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)
      all_img_name_vector.append(full_coco_image_path)
      all_captions.append(caption)

  # Shuffle captions and image_names together
  # Set a random state
  
  train_captions, img_name_vector = shuffle(all_captions,
                                        all_img_name_vector,random_state=1)
  #saving these lists saves us time while testing, rather than creating them                                      
  np.save('traincaptions_imagename.npy', [train_captions, img_name_vector])                                       
  
  # Taking a subset of the lists
  train_captions = train_captions[:total_size]
  img_name_vector = img_name_vector[:total_size]

  return img_name_vector,train_captions

def main():

  # The sample size from the total 4,14,803 is defined here
  total_size = 400000
  img_name_vector, train_captions, = preprocess(total_size)

  # The CNN encoder model is initialised here
  image_features_extract_model = image_features_model()

  # This function initially takes all the images at a batch size of 16
  # provides them to input to a CNN model, the activation output from the 
  # convolution layer is stored directly
  # run this function only once to create, rest of the time, comment this line
  # enc_len = batch_feature_processing(img_name_vector)  
  #print("Encoded length", enc_len, "len", len(img_name_vector))
  
  # This function takes in the captions, preprocess them and
  # return a sequemce of numbers which represent sequence of words 
  # part of a vocabulary
  cap_vector, tokenizer, max_length = proc_caption(train_captions)

  
  # Create training and validation sets using an 80-20 split
  img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                      cap_vector,
                                                                      test_size=0.2,
                                                                      random_state=0)

  print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val),"\n")


  # Training parameters according to  system's configuration
  top_k = 10000
  BATCH_SIZE = 64
  BUFFER_SIZE = 1000
  embedding_dim = 256
  units = 512

  # vocabuary of words
  vocab_size = top_k + 1
  num_steps = len(img_name_train) // BATCH_SIZE
  # Shape of the vector extracted from InceptionV3 is (64, 2048)
  # These two variables represent that vector shape
  features_shape = 2048
  attention_features_shape = 64

  loss_plot = []

  # Tensorflow data pipeline, similar to the documented example
  # Taking tensor slices from both the image name list
  # and corresponding caption
  dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

  #Use map to load the numpy files in parallel
  dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # Shuffle and batch
  dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  
  #MODEL STARTS HERE
  encoder = CNN_Encoder(embedding_dim)
  decoder = RNN_Decoder(embedding_dim, units, vocab_size)

  #OPTIMIZER AND LOSSS
  optimizer = tf.keras.optimizers.Adam()
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

  # Very own loss function according to paper
  def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

  
  #CHECKPOINTS
  checkpoint_path = "./checkpoints/train400000"
  ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer = optimizer)
  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)


  start_epoch = 0
  if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    print("LATEST CHECKPOINT:", ckpt_manager.latest_checkpoint)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    

  @tf.function   # this declaration is important, without this you will see errors
  def train_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    # The newer tensorflow uses Gradient Tape to perform Back Propogation
    with tf.GradientTape() as tape:
        # final image features are generated
        features = encoder(img_tensor)

        # target is the sequence of caption word numbers, iterating through its length
        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            # no need to have the attention weights here
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            
            # cumulating the loss from every time step
            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))
    # taking all the trainable parameters
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    # The gradients are calculated in this step and updated
    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss
  
  # The total no of epochs
  EPOCHS = 30

  for epoch in range(start_epoch, EPOCHS):
    # Each epoch is recorded in time
    start = time.time()
    total_loss = 0

    # using the prebuilt tensorflow data pipeline dataset and iterating with each batch
    for (batch, (img_tensor, target)) in enumerate(dataset):
      # calaculating each batch loss
      batch_loss, t_loss = train_step(img_tensor, target)
      total_loss += t_loss

      if batch % 100 == 0:
        print ('Epoch {} Batch {} Loss {:.4f}'.format(
          epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    # saving the model checkpoints
    if epoch % 5 == 0:
      ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                        total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
  
  plt.plot(loss_plot)
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Loss Plot')
  plt.savefig('training100000.jpg')
  plt.show()
  

  


if __name__== "__main__":
  
  # calling thew main function
  main()





