from keras_inbuilt_tokenizer import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import json

class TokenizerWrap(Tokenizer):
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
	

	D O C U M E N T A T I O N
	keras.preprocessing.text.Tokenizer class with extra functionality
	inspired by neural-machine-tutorials from https://github.com/Hvass-Labs/TensorFlow-Tutorials
	the init function decides what to pass filter ,split parameters for the Tokenizer
	if the text is in form of python code filter==? as '?'and'$' are unused symbols in python
	and split == '$' as dest data is formatted in such a way
	
	'''

	def __init__(self,texts,padding,reverse=False,num_words=None,coded_text=False,otype='src'):
		if otype == 'src':
			self.stfilename = 'data/src_tok.json'
		elif otype == 'dest':
			self.stfilename = 'data/dest_tok.json'
		# Check for pretokenized file and load if exists	
		exists = os.path.isfile(self.stfilename)
			
		#self.create(texts,padding,reverse=reverse,num_words=num_words,coded_text=coded_text,otype=otype)	
	
		if exists == True:
			if coded_text == True:
				filters = '?'
				split = '$'
			elif coded_text == False:
				filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'	
				split = ' '
			Tokenizer.__init__(self, num_words=num_words, filters=filters, split=split ,lower=False )

			dicti = json.loads(open(self.stfilename, "r").read())

			self.lower = dicti['lower']
			self.word_index=dicti['word_index']
			self.oov_token=dicti['oov_token']
			self.max_tokens=dicti['max_tokens']
			self.tokens=dicti['tokens']
			self.index_to_word=dicti['index_to_word']
			self.char_level=dicti['char_level']
			self.num_tokens=dicti['num_tokens']
			self.word_counts=dicti['word_counts']
			self.split=dicti['split']
			self.filters=dicti['filters']
			self.word_docs=dicti['word_docs']
			self.index_docs=dicti['index_docs']
			self.index_word=dicti['index_word']
			self.document_count=dicti['document_count']
			self.num_words=dicti['num_words']
			self.tokens_padded=np.array(dicti['tokens_padded']).reshape(len(self.tokens),self.max_tokens)
			if coded_text == True:
				json.dump(self.index_to_word,open('data/vocab.json', 'w'))



		else:
			if coded_text == True:
				filters = '?'
				split = '$'
			elif coded_text == False:
				filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'	
				split = ' '

			Tokenizer.__init__(self, num_words=num_words, filters=filters, split=split ,lower=False )
			# tokenized file not found so creating class from scratch
			# Create the vocabulary from the texts.
			self.fit_on_texts(texts)

			# Create inverse lookup from integer-tokens to words.
			self.index_to_word = dict(zip(self.word_index.values(),self.word_index.keys()))
			# Convert all texts to lists of integer-tokens.
			# Note that the sequences may have different lengths.
			self.tokens = self.texts_to_sequences(texts)

			if reverse:
				# Reverse the token-sequences.
				self.tokens = [list(reversed(x)) for x in self.tokens]
				# Sequences that are too long should now be truncated
				# at the beginning, which corresponds to the end of
				# the original sequences.
				truncating = 'pre'
			else:
				# Sequences that are too long should be truncated
				# at the end.
				truncating = 'post'
			# The number of integer-tokens in each sequence.
			self.num_tokens = [len(x) for x in self.tokens]
			# Max number of tokens to use in all sequences.
			# We will pad / truncate all sequences to this length.
			# This is a compromise so we save a lot of memory and
			# only have to truncate maybe 5% of all the sequences.
			self.max_tokens = np.mean(self.num_tokens)+ 2 * np.std(self.num_tokens)
			self.max_tokens = int(self.max_tokens)
			# Pad / truncate all token-sequences to the given length.
			# This creates a 2-dim numpy matrix that is easier to use.
			self.tokens_padded = pad_sequences(self.tokens,maxlen=self.max_tokens,padding=padding,truncating=truncating)

	def token_to_word(self, k):
		"""Lookup a single word from an integer-token.if integer token not available get value of nearest key numerically"""
		d = self.index_to_word
		return d[str(max(key for key in map(int, d.keys()) if key <= k))]
		 
	def text_to_tokens(self, text, reverse=False, padding=False):
		"""
		Convert a single text-string to tokens with optional
		reversal and padding.
		"""

		# Convert to tokens. Note that we assume there is only
		# a single text-string so we wrap it in a list.
		tokens = self.texts_to_sequences([text])
		tokens = np.array(tokens)
		if reverse:
			# Reverse the tokens.
			tokens = np.flip(tokens, axis=1)
			# Sequences that are too long should now be truncated
			# at the beginning, which corresponds to the end of
			# the original sequences.
			truncating = 'pre'
		else:
			# Sequences that are too long should be truncated
			# at the end.
			truncating = 'post'
		if padding:
			# Pad and truncate sequences to the given length.
			tokens = pad_sequences(tokens,maxlen=self.max_tokens,padding='pre',truncating=truncating)
		return tokens

	def dejsonify(self,ipclass):
		"""Function to convert back np-arrays converted to normal arrays for storing in json format"""
		for key , value in ipclass.iteritems():
			if key =='tokens_padded':
				ipclass[key] = np.array(value)		
		return ipclass		