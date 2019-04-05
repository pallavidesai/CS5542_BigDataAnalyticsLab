# This python file takes SBU dataset and perform Tokenization, Lemmatization on it.
# Filtering out the dataset with my traffic keywords and writing to the output text files.

import nltk
import numpy as np
import linecache
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1

# Download all stop words in English
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# Initializing Lemmatizer
lemmatizer = WordNetLemmatizer();
linecounter = 0
signalcount = exitcount = parkingcount =signcount = stopcount = 0
# Opening text files to put output image and captions
Output_caption = open('output_caption.txt','a')
Output_image = open('output_image.txt','a')
# These are punctuations we are removing from out captions
punctuation = "?:!.,;"

# Reading  SBU Caption data set text file
file  = open('SBU_captioned_photo_dataset_captions.txt').read()
with open('SBU_captioned_photo_dataset_captions.txt') as f:
    for line in f:
        linecounter = linecounter + 1
        # Tokenization
        token = word_tokenize(line)
        # Remove stop words and punctuations
        for word in token:
            if word in punctuation:
                token.remove(word)
            if word in stop_words:
                token.remove(word)
        # Getting rid of repetitive words 
        token = list(set(token))
        for word in token:
            #Lemmatization
            lemmit = lemmatizer.lemmatize(word)
            # Comparing with keywords
            if 'signal' == lemmit:
                signalcount=signalcount+1
                line1 = linecache.getline('SBU_captioned_photo_dataset_urls.txt', linecounter)
                # Writing to output files
                Output_caption.writelines(line)
                Output_image.writelines(line1)
            if 'exit' == lemmit:
                exitcount=exitcount+1
                line1 = linecache.getline('SBU_captioned_photo_dataset_urls.txt', linecounter)
                # Writing to output files
                Output_caption.writelines(line)
                Output_image.writelines(line1)
            if 'parking' == lemmit:
                parkingcount=parkingcount+1
                line1 = linecache.getline('SBU_captioned_photo_dataset_urls.txt', linecounter)
                # Writing to output files
                Output_caption.writelines(line)
                Output_image.writelines(line1)
            if 'sign' == lemmit:
                signcount=signcount+1
                line1 = linecache.getline('SBU_captioned_photo_dataset_urls.txt', linecounter)
                # Writing to output files
                Output_caption.writelines(line)
                Output_image.writelines(line1)
            if 'stop' == lemmit:
                stopcount=stopcount+1
                line1 = linecache.getline('SBU_captioned_photo_dataset_urls.txt', linecounter)
                # Writing to output files
                Output_caption.writelines(line)
                Output_image.writelines(line1)

# Total number of data obtained
total_used = signalcount+signcount+stopcount+parkingcount+exitcount

# Plot graph using matplot lib
objects = ('Signal', 'Sign', 'Stop','Parking','Exit')
y_pos = np.arange(len(objects))
performance = [signalcount,signcount,stopcount,parkingcount,exitcount]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.show()

plt1.scatter(linecounter, total_used, color='yellow')  # plotting the initial datapoints
plt1.plot(linecounter, total_used, color='blue', linewidth=3)  # plotting the line made by linear regression
plt1.show()











