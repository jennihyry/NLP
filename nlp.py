
#tags in the NLTK POS are found here: https://www.guru99.com/pos-tagging-chunking-nltk.html

#In corpus linguistics, part-of-speech tagging (POS tagging or PoS tagging or POST),
#also called grammatical tagging is the process of marking up a word in a text (corpus)
#as corresponding to a particular part of speech, based on both its definition and its context.

#a text corpus is a large body of text.

#Brown Corpus	Francis, Kucera	15 genres, 1.15M words, tagged, categorized
#Brown corpus source: http://korpus.uib.no/icame/brown/bcm.html
#NLTK data: https://www.nltk.org/nltk_data/
#NLTK corpus howto: https://www.nltk.org/howto/corpus.html

import nltk
from nltk import *
from nltk.corpus import brown
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from poem import poem1
import numpy as np
from collections import Counter
from preprocess import preprocess, lemmatize
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from scipy.optimize import curve_fit

processedBrown = preprocess(brown.words()) #used for assignment 1 and 2
processedPoem = preprocess(word_tokenize(poem1)) #used for assignment 5

def assignment_1(processedBrown):
	'''Frequency of words and POS tags in the corpus '''
	
    #*****************************************
	#Word part of assignment 1
    #*****************************************
	word = []
	count = []
	tag_list = []
	tag_x = []
	tag_count = []
	
	fdist = FreqDist(processedBrown)
	print(fdist)
	words = fdist.most_common(50)
	print(words)
	#fdist.plot(50) #Would plot the 50 most common words

	for i in range(0, len(words)):
		word.append(words[i][0])
		count.append(words[i][1])
	
	#Plotting
	plt.plot(word, count)
	plt.title("Frequency of words in pre-processed corpus")
	plt.xticks(word, labels = word, rotation = 'vertical')
	plt.ylabel('Count of words')
	plt.xlabel('Word')
	plt.show()

	np.savetxt('assignment1_filtered_corpus_data.csv', np.c_[word, count], delimiter = ',', fmt = '%s')
	
    #*****************************************
	#Tag part of assignment 1
	#*****************************************
	tags = nltk.pos_tag(processedBrown)
	fdist_tag = FreqDist(tags)
	fdist_tag_most_common = fdist_tag.most_common()

	"""
	for i in range(0, len(fdist_tag_most_common)):
		tag_x.append(fdist_tag_most_common[i][0])
		
		tag_count.append(fdist_tag_most_common[i][1])
	"""

	tag_counts = Counter(tag for word, tag in tags)

	for i, j in tag_counts.items():
		tag_list.append((i, j))
	tag_list.sort(key=lambda x: (-x[1], x[0]))
	[t[1] for t in tag_list]

	#Appending the tags and counts into two different lists
	for i in range(0, len(tag_list)):
		tag_x.append(tag_list[i][0])
		tag_count.append(tag_list[i][1])


	np.savetxt('assignment1_POS_filtered_data.csv', np.c_[tag_x, tag_count], delimiter = ',', fmt = '%s')

	plt.plot(tag_x, tag_count)
	plt.title("Frequency of POS-tags in pre-processed corpus")
	plt.xticks(tag_x, labels = tag_x, rotation = 'vertical')
	plt.ylabel('Count of tags')
	plt.xlabel('Tag')
	plt.show()	
    
def assignment_2(processedPoem):
    '''Fequency of words and POS tags in the poem'''
	word_list = []
	count_list = []
	tag_list = []
	tag_x = []
	tag_count = []
	lemmatized_list = []

	wnl = WordNetLemmatizer()
	non_processed_poem = word_tokenize(poem1)

	for word in processedPoem:
		lemmatized_list.append(wnl.lemmatize(word))

	lemmatized_pos_tags = nltk.pos_tag(lemmatized_list)
	print("lemmatized_pos_tags: ", lemmatized_pos_tags)
	tag_counts = Counter(tag for lemmatized_list, tag in lemmatized_pos_tags)

	for i, j in tag_counts.items():
		tag_list.append((i, j))
	tag_list.sort(key=lambda x: (-x[1], x[0]))
	[t[1] for t in tag_list]

	for i in range(0, len(tag_list)):
		tag_x.append(tag_list[i][0])
		tag_count.append(tag_list[i][1])

	#Plotting the tag count of the poem
	"""
	plt.plot(tag_x, tag_count)
	plt.title("Frequency of POS-tags in unprocessed poem")
	plt.xticks(tag_x, labels = tag_x, rotation = 'vertical')
	plt.ylabel('Count of tags')
	plt.xlabel('Tag')
	plt.show()	
	"""
	
	fdist = FreqDist(lemmatized_list)
	most_common_lemmatized = fdist.most_common(50)


	#SAVING THE LEMMATIZED WORDS IN A DATABASE
	for i in range(0, len(most_common_lemmatized)):
		word_list.append(most_common_lemmatized[i][0])
		count_list.append(most_common_lemmatized[i][1])
		
	np.savetxt('assignment2_lemmatized_words.csv', np.c_[word_list, count_list], delimiter = ',', fmt = '%s')
	
	#Plotting the word frequency of the poem
	plt.plot(word_list, count_list)
	plt.title("Frequency of words in processed poem")
	plt.xticks(word_list, labels = word_list, rotation = 'vertical')
	plt.ylabel('Count of words')
	plt.xlabel('Word')
	plt.show()
	
	
	#LAST 45 WORDS THAT ARE MENTIONED ONLY ONCE
	last_words = FreqDist(dict(fdist.most_common()[-45:]))
	print(last_words)
	last_words.plot()
	
		
	#POS-tag part of the assignment#
	
	#SAVING THE POS TAGS INTO A DATABASE
	for i, j in tag_counts.items():
		tag_list.append((i, j))

	tag_list.sort(key=lambda x: (-x[1], x[0]))
	[t[1] for t in tag_list]

	for i in range(0, len(tag_list)):
		tag_x.append(tag_list[i][0])
		tag_count.append(tag_list[i][1])

	np.savetxt('assignment2_lemmatized_tags.csv', np.c_[tag_x, tag_count], delimiter = ',', fmt = '%s')
	
	"""
	#THE PLOT FOR MOST COMMON POS TAGS
	print("Tag_counts: ", tag_counts)
	plt.plot(tag_x, tag_count)
	plt.xticks(tag_x, labels = tag_x, rotation = 'vertical')
	plt.ylabel('Count of tags')
	plt.xlabel('Tag')
	plt.show()
	"""

def assignment_3():
    '''Histogram of matching words in poem and corpus by percentage'''
	#CORPUS
	corpus_word = []
	corpus_word_count = []
	freq_count_pair = []

	#POEM
	poem_word = []
	poem_word_count = []
	poem_freq_count_pair = []
	
	#CORPUS
	with open('/Users/Patrick/Desktop/koulu/Python/NLP/assignment1_filtered_corpus_data.csv') as corpus_filtered_word_freq:
		corpus_csv_reader = csv.reader(corpus_filtered_word_freq, delimiter = ',')
		for i, j in corpus_csv_reader:
			freq_count_pair.append((i, j))
			corpus_word.append(i)
			corpus_word_count.append(j)

		#print("corpus_word: corpus_word_count: ", corpus_word, corpus_word_count)

	#POEM
	with open('/Users/Patrick/Desktop/koulu/Python/NLP/assignment2_lemmatized_words.csv') as poem_filtered_word_freq:
		poem_csv_reader = csv.reader(poem_filtered_word_freq, delimiter = ',')
		for i, j in poem_csv_reader:
			poem_freq_count_pair.append((i, j))
			poem_word.append(i)
			poem_word_count.append(j)

		#print("poem_word: poem_word_count: ", poem_word, poem_word_count)
	
	#CORPUS
	int_corpus_word_count_map = map(int, corpus_word_count)
	int_corpus_word_count = list(int_corpus_word_count_map)
	max_corpus_word_count = max(int_corpus_word_count)

	for i in range(0, len(corpus_word_count)):
		int_corpus_word_count[i] = int_corpus_word_count[i]/ max_corpus_word_count

	#print(int_corpus_word_count)

	#POEM
	int_poem_word_count_map = map(int, poem_word_count)
	int_poem_word_count = list(int_poem_word_count_map)
	max_poem_word_count = max(int_poem_word_count)


	for i in range(len(poem_word_count)):
		int_poem_word_count[i] = int_poem_word_count[i]/max_poem_word_count

	#print(int_poem_word_count)

	
	l=0
	p=0
	b=1
	n=5
	for k in range(6):
		print("Corpus top word frequencies {} - {}".format(b, n))
		for l in range(l, (l+5)):
			print("Corpus Word | Count: ", corpus_word[l], "|" , int_corpus_word_count[l])
		print("Poem top word frequencies {} - {}".format(b, n))
		for p in range(p, (p+5)):
			print("Poem Word | Count: ", poem_word[p], "|" , int_poem_word_count[p])
		l+=5
		b+=5
		n+=5
		print("")
	


	#save to csv and sort in decreasing order
	corpusFrame = pd.DataFrame({'Word': corpus_word, 'Frequency': int_corpus_word_count})
	corpusFrame = corpusFrame.sort_values(by= ['Frequency'], ascending=False)
	#corpusFrame.to_csv("assignment5_brown_data.csv")

	poemFrame = pd.DataFrame({'Word': poem_word, 'Frequency': int_poem_word_count})
	poemFrame = poemFrame.sort_values(by= ['Frequency'], ascending=False)
	#poemFrame.to_csv("assignment5_poem_data.csv")

	#matching bigrams and histogram: 
	percentages = []
	print("going in loop")
	for i in range(0, 30):
		slice1 = corpusFrame.iloc[(i*5):((i+1)*5)]
		slice2 = poemFrame.iloc[(i*5):((i+1)*5)]
		slice1 = slice1["Word"]
		slice2 = slice2["Word"]

		merged = slice1.append(slice2)
		merged = merged.duplicated()
		true_count = (merged.sum()/5)*100

		percentages.append(true_count)


	print(percentages)

	percentages_plot = []

	for i in range(0, len(percentages)):
		if len(percentages_plot) > 4:
			break
		percentages_plot.append(percentages[i])

	print(percentages_plot)
	y = [1, 2, 3, 4, 5]
	fig, ax = plt.subplots()

	plt.bar([1, 2, 3, 4, 5], percentages_plot)
	
	ax = plt.gca()
	ax.set_ylim([0, 100])
	ax.set_title("Histogram of similarities between Corpus and Poem words")
	ax.set_xticks(y)
	ax.set_xticklabels(labels = ["WORDS 1-5", "WORDS 6-10", "WORDS 11-15", "WORDS 16-20", "WORDS 21-25"])
	ax.set_xlabel("Top 1-25 words")
	ax.set_ylabel("Similarity of words (%)")


	plt.show()
	


def assignment_4():
'''Histogram of matching POS tags in poem and corpus by percentage'''
	#CORPUS
	corpus_tag = []
	corpus_tag_count = []
	corpus_tag_count_pair = []

	#POEM
	poem_tag = []
	poem_tag_count = []
	poem_tag_freq_count_pair = []
	
	#CORPUS
	with open('C:\koulu\NLP\projekti\assignment1_POS_filtered_data.csv') as corpus_filtered_tag_freq:
		corpus_csv_reader = csv.reader(corpus_filtered_tag_freq, delimiter = ',')
		for i, j in corpus_csv_reader:
			corpus_tag_count_pair.append((i, j))
			corpus_tag.append(i)
			corpus_tag_count.append(j)

		#print("corpus_tag: corpus_tag_count: ", corpus_tag, corpus_tag_count)

	#POEM
	with open('C:\koulu\NLP\projekti\assignment2_lemmatized_tags.csv') as poem_filtered_tag_freq:
		poem_csv_reader = csv.reader(poem_filtered_tag_freq, delimiter = ',')
		for i, j in poem_csv_reader:
			poem_tag_freq_count_pair.append((i, j))
			poem_tag.append(i)
			poem_tag_count.append(j)

	
	#CORPUS
	int_corpus_tag_count_map = map(int, corpus_tag_count)
	int_corpus_tag_count = list(int_corpus_tag_count_map)
	max_corpus_tag_count = max(int_corpus_tag_count)

	for i in range(0, len(corpus_tag_count)):
		int_corpus_tag_count[i] = int_corpus_tag_count[i] / max_corpus_tag_count

	#POEM
	int_poem_tag_count_map = map(int, poem_tag_count)
	int_poem_tag_count = list(int_poem_tag_count_map)
	max_poem_tag_count = max(int_poem_tag_count)

	for i in range(len(poem_tag_count)):
		int_poem_tag_count[i] = int_poem_tag_count[i] / max_poem_tag_count
	
	l=0
	p=0
	b=1
	n=5
	for k in range(6):
		try:
			print("Corpus top tag frequencies {} - {}".format(b, n))
			for l in range(l, (l+5)):
				print("Corpus tag | Count: ", corpus_tag[l], "|" , int_corpus_tag_count[l])
				
			print("Poem top tag frequencies {} - {}".format(b, n))
			for p in range(p, (p+5)):
				print("Poem tag | Count: ", poem_tag[p], "|" , int_poem_tag_count[p])
				
			l+=5
			b+=5
			n+=5
			print("")
		except IndexError:
			continue
	


	#save to csv and sort in decreasing order
	corpusFrame = pd.DataFrame({'tag': corpus_tag, 'Frequency': int_corpus_tag_count})
	corpusFrame = corpusFrame.sort_values(by= ['Frequency'], ascending=False)
	#corpusFrame.to_csv("assignment5_brown_data.csv")

	poemFrame = pd.DataFrame({'tag': poem_tag, 'Frequency': int_poem_tag_count})
	poemFrame = poemFrame.sort_values(by= ['Frequency'], ascending=False)
	#poemFrame.to_csv("assignment5_poem_data.csv")

	#matching bigrams and histogram: 
	percentages = []
	print("going in loop")
	for i in range(0, 30):
		slice1 = corpusFrame.iloc[(i*5):((i+1)*5)]
		slice2 = poemFrame.iloc[(i*5):((i+1)*5)]
		slice1 = slice1["tag"]
		slice2 = slice2["tag"]

		merged = slice1.append(slice2)
		merged = merged.duplicated()
		true_count = (merged.sum()/5)*100

		percentages.append(true_count)


	print(percentages)

	percentages_plot = []

	for i in range(0, len(percentages)):
		if percentages[i] > 0:
			percentages_plot.append(percentages[i])

	y = [1, 2, 3, 4, 5]
	fig, ax = plt.subplots()

	plt.bar([1, 2, 3, 4, 5], percentages_plot)
	
	ax = plt.gca()
	ax.set_ylim([0, 100])
	ax.set_title("Histogram of similarities between Corpus and Poem POS-tags")
	ax.set_xticks(y)
	ax.set_xticklabels(labels = ["TAGS 1-5", "TAGS 6-10", "TAGS 11-15", "TAGS 16-20", "TAGS 21-25"])
	ax.set_xlabel("Top 1-25 POS tags")
	ax.set_ylabel("Similarity of POS tags (%)")

	plt.show()

def assignment_5(corpus, poem):
    '''Find matching bigrams in corpus and poem'''
    corpus_bigrams = []
    corpus_count = []
    poem_bigrams_list = []
    poem_count = []
    
    
    #get the frequency distributions
    brown_bigrams = list(bigrams(corpus))         #tokenize all bigrams
    poem_bigrams = list(bigrams(poem))
    fdist_corpus = FreqDist(brown_bigrams)         
    fdist_poem = FreqDist(poem_bigrams)
    
    #differentiating the bigrams and frequencies for excel
    for bigram, number in fdist_corpus.items():
        corpus_bigrams.append(bigram)
        corpus_count.append(number)
        
    for bigramP, numberP in fdist_poem.items():
        poem_bigrams_list.append(bigramP)
        poem_count.append(numberP)
        
    
    #save to csv and sort in decreasing order
    corpusFrame = pd.DataFrame({'Bigram': corpus_bigrams, 'Frequency': corpus_count})
    corpusFrame = corpusFrame.sort_values(by= ['Frequency'], ascending=False)
    corpusFrame.to_excel("assignment5_brown_data.xlsx")
    
    poemFrame = pd.DataFrame({'Bigram': poem_bigrams_list, 'Frequency': poem_count})
    poemFrame = poemFrame.sort_values(by= ['Frequency'], ascending=False)
    poemFrame.to_excel("assignment5_poem_data.xlsx")
    
    #matching bigrams and histogram: 
    percentages = []
    
    for i in range(0, 30):
        slice1 = corpusFrame.iloc[(i*5):((i+1)*5)]
        slice2 = poemFrame.iloc[(i*5):((i+1)*5)]
        slice1 = slice1["Bigram"]         
        slice2 = slice2["Bigram"]
        
        merged = slice1.append(slice2)
        merged = merged.duplicated()
        true_count = (merged.sum()/5)*100
        percentages.append(true_count) 
    
    plt.hist(percentages, align='left')
    plt.title('Histogram of V1, .. , V30')
    plt.show()
    
def countSenses(word):
    '''calculates the number of senses a word has using wordnet'''
    return len(wordnet.synsets(word))


def assignment_6(poem):
    '''average sense per line'''
    averageSenses = []
    wordSenseDataFrame = pd.DataFrame(columns = ['Word', 'Senses'])
    
    #split the unprocessed poem into lines
    lines = poem.split('\\')
    
    #number of senses
    lineList = []
    for line in lines:
        senses = []
        words = []
        line = preprocess(word_tokenize(line))      #tokenize and preprocess
        line = lemmatize(line)                      #lemmatized list
        lineList.append(line)
        for word in line:
            senses.append(countSenses(word))            #senses has senses for ONE LINE
            words.append(word)
            
        df = pd.DataFrame({'Word': words , 'Senses' : senses})
        wordSenseDataFrame = wordSenseDataFrame.append(df)
        try:
            averageSenses.append(sum(senses)/len(senses))   #averageSenses has avg for EVERY LINE
        except ZeroDivisionError:
            averageSenses.append(0)
        
    lineSenseDataFrame = pd.DataFrame({'Line' : lineList, 'Avg senses' : averageSenses})
    
    wordSenseDataFrame = wordSenseDataFrame.drop_duplicates(subset='Word')
    
    lineSenseDataFrame.to_excel("assignment6_lineSense.xlsx")
    wordSenseDataFrame.to_excel("assignment6_wordSense.xlsx")

def objective(x, a, b, c, d, e):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + e
    
def assignment_7():
    '''code for curve fitting https://machinelearningmastery.com/curve-fitting-with-python/'''
    #read excel file and axis values
    dataframe = pd.read_excel("assignment6_wordSense.xlsx", usecols = [2])
    freq = dataframe['Senses'].value_counts()
    freq = freq.sort_index(ascending=True)
    y_values = freq.tolist()
    x_values = np.arange(0,len(y_values),1)
    
    #curve fit
    popt, _ = curve_fit(objective, x_values, y_values)
    a, b, c, d, e = popt
    plt.scatter(x_values, y_values)
    x_line = np.arange(min(x_values), max(x_values), 1)
    y_line = objective(x_line, a, b, c, d, e)
    
    #plot curve fit
    plt.plot(x_line, y_line, '--', color='red')
    plt.title("Polynomial curve fitting")
    plt.show()
    
    #plot sense frequency
    freq.plot(title="Sense Frequency of individual words", xlabel="Sense count", ylabel="Frequency")
    plt.show()
    
def assignment_8(poem):
    '''lexical diversity in the poem. LD = (adjectives+adverbs)/verbs'''
    lines = poem.split('\\')    #list of strings
    ld = []
    tags = []
    
    for line in lines:
        line = preprocess(word_tokenize(line))
        adj = 0
        adv = 0
        verb = 0
        #calculate adjectives, adverbs and verbs
        tags = nltk.pos_tag(line)
        for item in tags:
            if item[1] == 'JJ' or item[1] == 'JJR' or item[1] == 'JJS':
                adj = adj+ 1
            elif item[1] == 'RB' or item[1] == 'RBR' or item[1]=='RBS':
                adv =adv + 1
            elif item[1] == 'VBG' or item[1] == 'VBD' or item[1]=='VBN' or item[1] == 'VBP' or item[1]=='VBZ':
                verb = verb+ 1
        try:
            ld.append((adj+adv)/verb)
        except ZeroDivisionError:
            ld.append((adj+adv)) #assume that every sentence has at least one verb
    lexDiv = pd.DataFrame({'Line': lines, 'Lexical diversity' : ld})
    lexDiv.to_excel("assignment8_lexdiv.xlsx")
    lexDiv.plot()
    plt.title("Lexical Diversity by lines")
    plt.show()
    
    #bins
    bins = []
    binsize = (lexDiv['Lexical diversity'].max() - lexDiv['Lexical diversity'].min())/10
    binsize = round(binsize, 2)
    bins.append(lexDiv['Lexical diversity'].min())
    
    for i in range(0,10):
        roundedBin = round((bins[i] + binsize) , 2)
        bins.append(roundedBin)
    
    #histogram
    plt.hist(lexDiv['Lexical diversity'].tolist() , bins)
    plt.title('Histogram of lexical diversity')
    plt.xlabel('Lexical diversity')
    plt.ylabel('Frequency')
    plt.show()
    
    
    
    
#assignment_1(processedBrown)
#assignment_2(processedPoem)
#assignment_3()
#assignment_4()
#assignment_5(processedBrown, processedPoem)
#assignment_6(poem1)
#assignment_7()
#assignment_8(poem1)

#print(brown.words())