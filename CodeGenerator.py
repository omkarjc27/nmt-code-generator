import sys
import os
import datetime
import platform

	
main_module = main_network.Network()
def generate(input_string,expt=expt):
	return main_module.generate(str(input_string),expt)

def train(batchsize,valsize):
	main_module.train(batchsize,valsize)

def get_batchsize():
	bs = 32
	try:
		bs = int(raw_input('Enter Batch Size(Default is 32):'))
	except (EOFError, ValueError):
		pass
	return bs

helpinfo = ' This is CodeGenerator help menu .\n Instructions \n  >>train \t\t\t Train the model.\n'
print(str('-'*60+'\nCodeGenerator v1.0 |NMTCodeGen| \nat ['+str(datetime.datetime.now())+']\nRunning on '+str(platform.system())+' '+str(platform.release())+'\nType "help" for more instructions & "quit" to exit\n'+'-'*60))

c_input = ''
while c_input != "quit":
	c_input = raw_input(">>")
	if c_input == "train":
		train(get_batchsize(),0.05)	
	elif c_input == "help":
		print(helpinfo)
	elif c_input == "quit":
		pass
	else:
		print("Predicted Code :\n"+str(generate(c_input,raw_input('Expected Code >>'))))		