import json
from data_tokenizer import TokenizerWrap

data_src = json.loads(open("data/src.json", "r").read())
['' if type(i) is None else i for i in data_src]
data_dest = json.loads(open("data/dest.json", "r").read())
['' if type(i) is None else i for i in data_dest]
num_words = 10000
tokenizer_src =TokenizerWrap(texts=data_src,padding='pre',reverse=True,num_words=num_words,coded_text=False) 
tokenizer_dest = TokenizerWrap(texts=data_dest,padding='post',reverse=False,num_words=num_words,coded_text=True)

def jsonify(ipclass):
	for key , value in ipclass.iteritems():
		print(key)
		if key =='tokens_padded':
			ipclass[key] = value.tolist()		
	return ipclass	

print("dest")
d_dict = jsonify(tokenizer_dest.__dict__)
print("src")
s_dict =jsonify(tokenizer_src.__dict__)

json.dump(d_dict,open('data/dest_tok.json', "w"))
json.dump(s_dict,open('data/src_tok.json', "w"))