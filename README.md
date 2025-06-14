import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from PIL import Image
import os

# Define your dataset and captions
captions = [...]  # Replace with your captions
image_data = [...]  # Replace with your image data

# Load pre-trained ResNet model without top (classification) layer
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from PIL import Image
import os

# Define your dataset and captions
captions = [...]  # Replace with your captions
image_data = [...]  # Replace with your image data

# Load pre-trained ResNet model without top (classification) layer
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Define the image captioning model
vocab_size = 10000  # Define your vocabulary size
embedding_dim = 256  # Define your embedding dimension
max_seq_length = 20  # Define your maximum sequence length

image_input = Input(shape=(224, 224, 3))
encoded_image = base_model(image_input)
flatten = tf.keras.layers.Flatten()(encoded_image)
dense = Dense(256, activation='relu')(flatten)

caption_input = Input(shape=(max_seq_length,))
embedding_layer = Embedding(vocab_size, embedding_dim, input_length=max_seq_length)(caption_input)
lstm_layer = LSTM(256)(embedding_layer)

decoder = tf.keras.layers.add([dense, lstm_layer])
output = Dense(vocab_size, activation='softmax')(decoder)

model = Model(inputs=[image_input, caption_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Print the model summary
model.summary()

# Tokenize
 Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
94765736/94765736 [==============================] - 1s 0us/step
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_3 (InputLayer)        [(None, 224, 224, 3)]        0         []                            
                                                                                                  
 resnet50 (Functional)       (None, 7, 7, 2048)           2358771   ['input_3[0][0]']             
                                                          2                                       
                                                                                                  
 input_4 (InputLayer)        [(None, 20)]                 0         []                            
                                                                                                  
 flatten (Flatten)           (None, 100352)               0         ['resnet50[0][0]']            
                                                                                                  
 embedding (Embedding)       (None, 20, 256)              2560000   ['input_4[0][0]']             
                                                                                                  
 dense (Dense)               (None, 256)                  2569036   ['flatten[0][0]']             
                                                          8                                       
                                                                                                  
 lstm (LSTM)                 (None, 256)                  525312    ['embedding[0][0]']           
                                                                                                  
 add (Add)                   (None, 256)                  0         ['dense[0][0]',               
                                                                     'lstm[0][0]']                
                                                                                                  
 dense_1 (Dense)             (None, 10000)                2570000   ['add[0][0]']                 
                                                                                                  
==================================================================================================
Total params: 54933392 (209.55 MB)
Trainable params: 31345680 (119.57 MB)
Non-trainable params: 23587712 (89.98 MB)
__________________________________________________________________________________________________
