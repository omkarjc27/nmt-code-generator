from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from data_tokenizer import TokenizerWrap
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import os
import sys
import json


class Network(object):

	def __init__(self):
		'''
			L I C E N S E
		The MIT License (MIT) 
		Copyright (c) 2016 by Magnus Erik Hvass Pedersen
		Permission is hereby granted, free of charge, to any person obtaining a copy
		of this software and associated documentation files (the "Software"), to deal
		in the Software without restriction, including without limitation the rights
		to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
		copies of the Software, and to permit persons to whom the Software is
		furnished to do so, subject to the following conditions:
		The above copyright notice and this permission notice shall be included in all
		copies or substantial portions of the Software.
		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
		IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
		FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
		AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
		LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
		OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
		SOFTWARE.
		Original Repository :- https://github.com/Hvass-Labs/TensorFlow-Tutorials
		'''
		'''
		D O C U M E N T A T I O N
		Data extracted from "CoNaLa-Corpus(v1.1)" [http://www.phontron.com/download/conala-corpus-v1.1.zip]
		src =>> conala-corpus/conala-test.json/rewritten_intent + conala-corpus/conala-train.json/rewritten_intent
		dest =>> conala-corpus/conala-test.json/snippet + conala-corpus/conala-train.json/snippet
		'''
		print('Initiating CodeGenerator...')
		self.data_strt =0
		self.data_end = 40000
		self.data_src = json.loads(open("data/src.json", "r").read())
		['' if type(i) is None else i for i in self.data_src]
		self.data_dest = json.loads(open("data/dest.json", "r").read())
		['' if type(i) is None else i for i in self.data_dest]
		print('og data loaded')
		self.mark_start = ' sss777s777s ' #	start words for all sequences
		self.mark_end = ' ee7e7e7e ' #	end words for all sequences
		self.num_words = 10000 #	no of max words in vocabulary
		self.tokenizer_src =TokenizerWrap(texts=self.data_src,padding='pre',reverse=True,num_words=self.num_words,coded_text=False,otype='src') 
		self.tokenizer_dest = TokenizerWrap(texts=self.data_dest,padding='post',reverse=False,num_words=self.num_words,coded_text=True,otype='dest')
		print("tokenized")
		self.tokens_src = self.tokenizer_src.tokens_padded
		self.tokens_dest = self.tokenizer_dest.tokens_padded
		self.token_start = self.tokenizer_dest.word_index[self.mark_start.strip()]
		self.token_end = self.tokenizer_dest.word_index[self.mark_end.strip()]
		self.decoder_input_data = self.tokens_dest[:,:-1]
		self.decoder_output_data = self.tokens_dest[:,1:]
		if self.decoder_input_data.shape != self.decoder_output_data.shape :
			print(self.decoder_input_data.shape)
			print(self.decoder_output_data.shape)
			sys.exit("Shapes Not Identical ...")

		self.embedding_size = 128  
		self.state_size = 512
		self.encoder_input_data = self.tokens_src[self.data_strt:self.data_end]
		self.encoder_input = Input(shape=(None, ), name='encoder_input') 
		self.encoder_embedding = Embedding(input_dim=self.num_words,output_dim=self.embedding_size,name='encoder_embedding')	
		print('model start')
		# ENCODER start------------------------
		self.e1 = LSTM(self.state_size, name='encoder_lstm1',return_sequences=True)# Encoder LSTM Layer 1
		self.e2 = LSTM(self.state_size, name='encoder_lstm2',return_sequences=True)# Encoder LSTM Layer 2
		self.e3 = LSTM(self.state_size, name='encoder_lstm3',return_sequences=False)# Encoder LSTM Layer 3
		self.encoder_output = self.connect_encoder()# Connection all alyers of encoder
		# DECODER start-------------------------
		self.decoder_initial_state = Input(shape=(self.state_size,),name='decoder_initial_state')
		self.decoder_input = Input(shape=(None, ), name='decoder_input')
		self.decoder_embedding = Embedding(input_dim=self.num_words,output_dim=self.embedding_size,name='decoder_embedding')
		self.d1 = LSTM(self.state_size, name='decoder_lstm1',return_sequences=True)# DECODER LSTM Layer 1
		self.d2 = LSTM(self.state_size, name='decoder_lstm2',return_sequences=True)# DECODER LSTM Layer 2
		self.d3 = LSTM(self.state_size, name='decoder_lstm3',return_sequences=True)# DECODER LSTM Layer 3
		self.decoder_dense = Dense(self.num_words,activation='linear',name='decoder_output')#used as activation function
		print('models created')
		# MODEL 1----------------------------------------------------
		self.decoder_output = self.connect_decoder(self.encoder_output)
		self.model_train = Model(inputs=[self.encoder_input, self.decoder_input],outputs=[self.decoder_output])
		# MODEL 2----------------------------------------------------
		self.model_encoder = Model(inputs=[self.encoder_input],outputs=[self.encoder_output])
		# MODEL 3----------------------------------------------------
		self.decoder_output = self.connect_decoder(initial_state=self.decoder_initial_state)
		self.model_decoder = Model(inputs=[self.decoder_input, self.decoder_initial_state],outputs=[self.decoder_output])
		print('models connected')

	def connect_encoder(self):
		net = self.encoder_input
		net = self.encoder_embedding(net)
		net = self.e1(net)
		net = self.e2(net)
		net = self.e3(net)
		encoder_output = net
		return encoder_output

	def connect_decoder(self,initial_state):
		net = self.decoder_input
		net = self.decoder_embedding(net)
		net = self.d1([net, initial_state, initial_state])
		net = self.d2([net, initial_state, initial_state])
		net = self.d3([net, initial_state, initial_state])
		# Connect the final dense layer that converts to
		# one-hot encoded arrays.
		decoder_output = self.decoder_dense(net)
		
		return decoder_output
	
	def sparse_cross_entropy(self,y_true, y_pred):
		#----------------------------------------------
		#---------------LOSS FUNCTION------------------
		#----------------------------------------------
		return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

	def train(self,batch_size,validation_split):
		#----------------------------------------------
		#---------------- TRAINING --------------------
		#----------------------------------------------
		print("Compiling Model ...")
		optimizer = RMSprop(lr=1e-3)
		decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
		self.model_train.compile(optimizer=optimizer,loss=self.sparse_cross_entropy,target_tensors=[decoder_target])

		#----------------------------------------------
		#--------------Save CHECKPOINTS----------------
		#----------------------------------------------
		print("Searching Old Checkpoints ...")
		self.path_checkpoint = 'checkpoint_code_generator_training.keras'
		self.callback_checkpoint = ModelCheckpoint(filepath=self.path_checkpoint,monitor='val_loss',verbose=1,save_weights_only=True,save_best_only=True)
		self.callback_tensorboard = TensorBoard(log_dir='./pre-trained/',histogram_freq=0,write_graph=False)
		self.callback_early_stopping = EarlyStopping(monitor='val_loss',patience=3, verbose=1)
		self.callbacks = [self.callback_early_stopping,self.callback_checkpoint,self.callback_tensorboard]

		#----------------------------------------------
		#-------------LOAD CHECKPOINTS-----------------
		#----------------------------------------------

		try:
			self.model_train.load_weights(self.path_checkpoint)
		except Exception as error:
			print("No Old Checkpoints Found ... ")


		#	creating dictionaries
		self.x_data = {'encoder_input': self.encoder_input_data,'decoder_input': self.decoder_input_data}
		self.y_data = {'decoder_output': self.decoder_output_data}

		#----------------------------------------------------------------------------------------------------------------
		print("Training...")
		self.model_train.fit(x=self.x_data,y=self.y_data,batch_size=batch_size,epochs=10,validation_split=validation_split,callbacks=self.callbacks,verbose=1)
		
		#----------------------------------------------------------------------------------------------------------------		

	def generate(self,input_text,expt=None):
		input_tokens = self.tokenizer_src.text_to_tokens(text=input_text,reverse=True,padding=True)
		initial_state = self.model_encoder.predict(input_tokens)
		max_tokens = self.tokenizer_dest.max_tokens
		shape = (1, max_tokens)
		decoder_input_data = np.zeros(shape=shape, dtype=np.int)
		# The first input-token is the special start-token for '###START###'.
		token_int = self.token_start
		# Initialize an empty output-text.
		output_text = ''
		# Initialize the number of tokens we have processed.
		count_tokens = 0
		# While we haven't sampled the special end-token for '###END###'
		# and we haven't processed the max number of tokens.
		print(len(self.tokenizer_dest.index_to_word))
		print(type(self.tokenizer_dest.index_to_word))
		while token_int != self.token_end and count_tokens < max_tokens:
			decoder_input_data[0, count_tokens] = token_int
			x_data = {'decoder_initial_state': initial_state,'decoder_input': decoder_input_data}
			decoder_output = self.model_decoder.predict(x_data)
			token_onehot = decoder_output[0, count_tokens, :]
			token_int = np.argmax(token_onehot)
			sampled_word = self.tokenizer_dest.token_to_word(token_int)
			output_text += sampled_word
			count_tokens += 1
		output_tokens = decoder_input_data[0]
		return(output_text)