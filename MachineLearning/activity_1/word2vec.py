
import gzip
import gensim 
import logging
logging.basicConfig(level=logging.DEBUG)

def read_input(input_file):
	"""This method reads the input file which is in gzip format"""
	
	logging.info("reading file {0}...this may take a while".format(input_file))
	
	with gzip.open (input_file, 'rb') as f:
		for i, line in enumerate (f): 

			if (i%10000==0):
				logging.info ("read {0} reviews".format (i))
			# do some pre-processing and return a list of words for each review text
			yield gensim.utils.simple_preprocess (line)

a=10
print (a)

documents = list (read_input ("reviews_data.txt.gz"))
print ("Done reading data file")

model = gensim.models.Word2Vec (documents, size=100, window=10, min_count=2, workers=10)
print ("Training model")
model.train(documents,total_examples=len(documents),epochs=10)
print ("Model trained")

w1 = "dirty"
print (model.wv.most_similar (positive=w1))

print (model['dirty'])

w1 = "france"
print (model.wv.most_similar (positive=w1,topn=6))

w1 = "dirty"
print (model.wv.most_similar (positive=w1))

w1 = "dirty"
w2  ="clean"
print (model.wv.similarity (w1,w2))

w1 = "red"
print (model.wv.most_similar (positive=w1,topn=6))

import pickle

fileObject = open("modelogensim",'wb')

pickle.dump(model,fileObject)

fileObject.close()

fileObject = open("modelogensim",'r')  
# load the object from the file into var b
b = pickle.load(fileObject)

print (b.wv.most_similar (positive=w1,topn=6))

a = b.wv.most_similar(positive=w1,topn=6)

print(a[1][1])
