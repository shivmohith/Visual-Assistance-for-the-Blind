from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Image_Captioning.model import *
from Image_Captioning.batch_feature import image_features_model, load_image
from Image_Captioning.process_caption import proc_caption
import cv2
import pickle as pk
label_encoder = pk.load(open('label_encoder_lstm.sav','rb'))

'''
VQA Model configurations
'''

img_dim                  =     4096
word2vec_dim             =      300
#max_len                 =       30 # Required only when using Fixed-Length Padding
num_hidden_nodes_mlp     =     1024
num_hidden_nodes_lstm    =      512
num_layers_mlp           =        3
num_layers_lstm          =        3
dropout                  =       0.5
activation_mlp           =     'tanh'

'''
Image captioning model configurations
'''

total_size =100000
top_k = 5000
embedding_dim = 256
units = 512
vocab_size = top_k + 1 
attention_features_shape = 64



def load_image_features_model():
    ''' 
    Returns the image features extracting model
    '''

    base_model = tf.keras.applications.VGG16(weights='imagenet')
    image_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return image_model

image_model = load_image_features_model()

def get_image_features(image_model, image_path):
    ''' Returns numpy array
    Extracts the image features using VGG16

    Args:
        image_model: image features extracting model
        image_path: location of image
    '''

    img = cv2.imread(image_path)
    I = cv2.resize(img, (224, 224)) # - np.array((103.939, 116.779, 123.680), dtype=np.float32)
    x = image.img_to_array(I)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    image_features = image_model.predict(x)
    image_features = np.asarray(image_features, dtype=np.float32)
    return image_features

def get_question_features(question, nlp):
    ''' Returns numpy array
    Converts the question into an array of vectors using word2vec model

    Args:
        question: question in string format
        nlp: word2vec model
    '''

    tokens = nlp(question)
    question_features = np.zeros((1, len(tokens), 300))
    for j in range(len(tokens)):
        question_features[0,j,:] = tokens[j].vector
    return question_features

def load_vqa_model():
    '''
    Forms, compiles, and returns the VQA model
    '''

    image_model = tf.keras.Sequential([tf.keras.layers.Reshape(input_shape = (img_dim,), target_shape=(img_dim,), name='Feeding_image_vectors_size_4096')], name='Image_Model')
    image_model.add(tf.keras.layers.Dense(num_hidden_nodes_mlp, kernel_initializer='uniform', name = 'Image_MLP_Hidden_layer_size_1024'))
    image_model.add(tf.keras.layers.Activation('tanh', name='Image_MLP_Activation_tanh'))
    image_model.add(tf.keras.layers.Dropout(0.5, name='Image_MLP_Dropout_0.5'))

    language_model = tf.keras.Sequential([tf.keras.layers.LSTM(num_hidden_nodes_lstm,return_sequences=True, input_shape=(None, word2vec_dim), name='Feeding_question_vectors_to_LSTM_Layer_1'),
                                      tf.keras.layers.LSTM(num_hidden_nodes_lstm, return_sequences=True, name='LSTM_layer_2'),
                                      tf.keras.layers.LSTM(num_hidden_nodes_lstm, return_sequences=False, name='LSTM_layer_3')
                                      ], name = 'Language_Model')
    language_model.add(tf.keras.layers.Dense(num_hidden_nodes_mlp, kernel_initializer='uniform', name = 'Question_MLP_Hidden_layer_size_1024'))
    language_model.add(tf.keras.layers.Activation('tanh', name='Question_MLP_Activation_tanh'))
    language_model.add(tf.keras.layers.Dropout(0.5, name='Question_MLP_Dropout_0.5'))

    upper_lim = 1000 

    merged=tf.keras.layers.concatenate([language_model.output,image_model.output], axis =-1, name='Merging_language_model_and_image_model')

    model =tf.keras.Sequential(name='CNN_LSTM_Model')(merged)
    model = tf.keras.layers.Dense(num_hidden_nodes_mlp, kernel_initializer='uniform', name = 'Combined_MLP_Hidden_layer_size_1024')(model)
    model = tf.keras.layers.Activation('tanh', name='Combined_MLP_Activation_tanh')(model)
    model = tf.keras.layers.Dropout(0.5, name='Combined_MLP_Dropout_0.5')(model)
    model = tf.keras.layers.Dense(upper_lim, name='Fully_Connected_Output_layer_size_1000')(model)
    out =   tf.keras.layers.Activation("softmax", name='Softmax_Output_Probablities')(model)
    model = tf.keras.Model([language_model.input, image_model.input], out, name='LSTM_CNN_Model')
    model.load_weights('weights_cnn_lstm_v2/LSTM_1000classes_epoch_59.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def predict_answer(model, nlp, question, image_path):
    ''' Returns a string
    Gets the image features and questions features, and predicts the answer using VQA model

    Args:
        model: VQA model
        nlp: word2vec model
        question: question in string format
    ''' 

    image_features = get_image_features(image_model,image_path)
    question_features = get_question_features(question, nlp)
    input_data = [question_features, image_features]
    y_predict = model.predict(input_data, verbose=0)
    y_predict = np.argmax(y_predict,axis=1)
    return label_encoder.inverse_transform(y_predict)[0]

def load_image_caption_model():
    ''' Returns encoder model, decoder model image_features_extract_model, tokenizer and max length of caption
    Forms, compiles, and loads the image caption model
    '''

    train_captions,img_name_vector = np.load('./Image_Captioning/traincaption_imgname.npy')
    img_name_vector = img_name_vector[:total_size].tolist()
    train_captions = train_captions[:total_size].tolist()

    image_features_extract_model = image_features_model()
    
    _, tokenizer, max_length = proc_caption(train_captions)       

    #MODEL STARTS HERE
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    #OPTIMIZER 
    optimizer = tf.keras.optimizers.Adam()

    #CHECKPOINTS
    checkpoint_path = "./Image_Captioning/checkpoints/train100000"
    ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    return encoder, decoder, image_features_extract_model, tokenizer, max_length

def generate_caption(image_path, encoder, decoder, image_features_extract_model, tokenizer, max_length):
    ''' Returns string and tensorflow object
    Given an image, generates a caption

    Args:
        image_path: location of the image
        encoder: encoder model
        decoder: decoder model
        image_features_extract_model: model which extracts the image features
        tokenizer: tokenizer 
        max_length: maximum length of the caption
    '''

    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image_path)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
      predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

      attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

      predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
      result.append(tokenizer.index_word[predicted_id])

      if tokenizer.index_word[predicted_id] == '<end>':
          return result, attention_plot

      dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_image_caption_attention(result, attention_plot, image):
    '''
    Plots the attention over the image
    '''
    
    temp_image = np.array(Image.open(image))
    n=0
    fig = plt.figure(num=n, figsize=(120,120))

    len_result = len(result)
    j =0
    for l in range(len_result):
        #print("l is:", l, len_result, "len_result//6", len_result//6)
        temp_att = np.resize(attention_plot[l], (8, 8))
        if((l%10)==0 and l >0):
            #ax = fig.add_subplot(len_result//6, len_result//6, l+1)
            n+=1
            j =0
            fig = plt.figure( num=n,figsize=(120,120))
            ax = fig.add_subplot(3,4, j+1)
        else:
            ax = fig.add_subplot(3,4,j+1)
            j+=1
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
        plt.tight_layout(pad=50, w_pad=50.0, h_pad=50.0)
            
    plt.tight_layout(pad=50, w_pad=50.0, h_pad=50.0)
    plt.show()