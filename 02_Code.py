# Import Necessary Libraries

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as ps
from nltk import WordNetLemmatizer as wn

from nltk.tokenize import sent_tokenize , word_tokenize
import string
import re

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

from flask import Flask, jsonify, request
import sklearn.externals.joblib as jb

#############################################################

data = pd.read_csv('jobs_data.csv', index_col = 0)
data = data.drop(['industry'], axis = 1) # removing useless features 
# data = data.drop(['Unnamed: 0', 'industry'], axis = 1) # removing useless features 
data.head()

#############################################################

# what is the shape of the dataset?
print('the data has {} rows and has {} cols'.format(len(data) , len(data.columns)))

#############################################################


unduplicata_data = data.drop_duplicates(subset='title')
unduplicata_data.shape

#############################################################


stopwords = nltk.corpus.stopwords.words('english')
stopwords[:5]

#############################################################

# function for make some of NLP functions

# this func for title 

def clean_text_without_splitting(text):
    text = re.sub("[^A-Za-z]", " ", text.strip())                              # Remove non english words
    text = "".join([word for word in text if word not in string.punctuation]) # Removing puncituation
    tokenize = re.split('\W+' , text)                                        # Tokenization    
    text = [word.lower() for word in tokenize if word not in stopwords]     # Removing stopwords , lower case
    text = [word for word in text if len(word)]                            # Remove the empty string
    text = " ".join(word for word in text)                                # Make it as a sentence     
    return text

# this func for jobFunction 

def clean_text_with_splitting(text):
    text = re.sub("[^A-Za-z]", " ", text.strip())                              # Remove non english words
    text = "".join([word for word in text if word not in string.punctuation]) # Removing puncituation
    tokenize = re.split('\W+' , text)                                        # Tokenization    
    text = [word.lower() for word in tokenize if word not in stopwords]     # Removing stopwords , lower case
    text = [word for word in text if len(word)]                            # Remove the empty string    
    return text

#############################################################

data['clean_title'] = data['title'].apply(lambda x : clean_text_without_splitting(x)) # without splitting mean as string
data['clean_jobFunction'] = data['jobFunction'].apply(lambda x : clean_text_with_splitting(x)) # with splitting 

#############################################################


# delete the old cols
data = data.drop(['title', 'jobFunction'], axis=1)
data.head(10)

#############################################################

# this func return the similarity between q1 and q2

def jaccard_similarity_1(q1, q2):
    intersection = set(q1).intersection(set(q2))
    union = set(q1).union(set(q2))
    return len(intersection)/len(union)
    
# take each titel in clean title (q2) and make a similarity between q2 and q1(the input) and call jaccard_similarity_1
    
def jaccard_similarity_2(q1):
    sim=[]
    title_list = data['clean_title'] # second qurery for make jac_sim => q2    
    for i in range(len(title_list)):
        q2 = title_list[i]
        sim.append((jaccard_similarity_1(q1, q2), i))
    return sim


# take a list of tuples [(sim_num, index)] and sorted them descending by sim_num
# make a list of all indeces from high similarity to low 
# make a list of top titles and return them

def sorting_rev(list_of_sim_and_index):
    sorted_sim_list = sorted(list_of_sim_and_index , reverse= True)
    
    # make a list of high similarity of spicific title
    index_of_high_sim = []
    title_of_high_sim = []
    for i in range(len(sorted_sim_list)):
        index_of_high_sim.append(sorted_sim_list[i][1]) # sorted_sim_list => (sim, index)
    
    # here we iterate on list of indeces and take just 6 job function     
    flag = 0
    for i in range(len(index_of_high_sim)):
        if(flag):
            break
        for j in range(len(data['clean_jobFunction'][index_of_high_sim[i]])):
            if (data['clean_jobFunction'][index_of_high_sim[i]][j] not in title_of_high_sim and (len(data['clean_jobFunction'][index_of_high_sim[i]][j]) > 1)):
                title_of_high_sim.append(data['clean_jobFunction'][index_of_high_sim[i]][j])
                if(len(title_of_high_sim)> 6):
                    flag =1 
                    break
                
    
    sample_of_title_of_high_sim = title_of_high_sim[:6]
    return sample_of_title_of_high_sim

def print_it_as_atitle(list_of_job_title):
    for st in list_of_job_title:
        print(st.title())
#############################################################
    
    
    
def recommendation_job_functions(query):
    clean_query = clean_text_without_splitting(query)          # text preprocessing for input
    list_of_indx = jaccard_similarity_2(clean_query)           # return list of tuple for (sim, index)
    list_of_job_functions = sorting_rev(list_of_indx)          # return list of sorted indices
    # final_answer = print_it_as_atitle(list_of_job_functions) # print the final answer as a Title string
    return list_of_job_functions

#############################################################


q = "IOS Developer"

print(recommendation_job_functions(q))


# app
app = Flask(__name__)


# routes
@app.route('/', methods=['Get'])
def printTheJobFunction():
    list_of_job_funs = recommendation_job_functions(q)
    print('The job title is: ', q , "\n")
    # return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
    # return jsonify('The specific job functions of {} are : {}'.format(q, [t.title() for t in list_of_job_funs]))
    
    return jsonify({'The specific job functions of {} are'.format(q) :[t.title() for t in list_of_job_funs]})

@app.route('/lang', methods= ['Get'])
def returnAll():
    return jsonify({'Languages':languages})



if __name__ == '__main__':
    app.run(port = 2221, debug=True)
    
