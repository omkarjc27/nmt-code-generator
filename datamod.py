import json
import jsonlines

def initiate():
	with open('conala-corpus/conala-train.json','rb') as f:
		for item in json.load(f):
			snippet.append(purify(item['snippet']))
			intent.append(item['rewritten_intent'])		
	with open('conala-corpus/conala-train.json','rb') as f:
		for item in json.load(f):
			snippet.append(purify(item['snippet']))
			intent.append(item['rewritten_intent'])		

	with open('conala-corpus/conala-mined.jsonl','rb') as f:
		for item in jsonlines.Reader(f):
			snippet.append(purify(item['snippet']))
			intent.append(item['intent'])			

def purify(text):
	text = text.replace('!','$!$').replace('"','$"$').replace('#','$#$').replace('%','$%$').replace('&','$&$').replace('(','$($').replace(')','$)$').replace('*','$*$').replace('+','$+$').replace(',',',$').replace('-','$-$').replace('.','$.$').replace('/','$/$').replace(':','$:$').replace(';','$;$').replace('<','$<$').replace('=','$=$').replace('>','$>$').replace('?','$?$').replace('@','$@$').replace('[','$[$').replace('\\','$\\$').replace(']','$]$').replace('^','$^$').replace('_','$_$').replace('`','$`$').replace('{','${$').replace('|','$|$').replace('}','$}$').replace('~','$~$').replace('\t','$\t$').replace('\n','$\n$').replace("'", "$'$").replace(" ","$ $")
	text = 'sss777s777s$'+text+'$ee7e7e7e'
	return text

snippet = []
intent = []
initiate()
with open('data/src.json','w') as outfile:
    json.dump(intent, outfile)
with open('data/dest.json','w') as outfile:
    json.dump(snippet, outfile)
