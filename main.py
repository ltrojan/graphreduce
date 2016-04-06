'''

  _____                 _     _____          _                
  / ____|               | |   |  __ \        | |               
 | |  __ _ __ __ _ _ __ | |__ | |__) |___  __| |_   _  ___ ___ 
 | | |_ | '__/ _` | '_ \| '_ \|  _  // _ \/ _` | | | |/ __/ _ \
 | |__| | | | (_| | |_) | | | | | \ \  __/ (_| | |_| | (_|  __/
  \_____|_|  \__,_| .__/|_| |_|_|  \_\___|\__,_|\__,_|\___\___|
                  | |                                          
                  |_|   

 
 A graph-based text summarizer for easier consumption of product reviews

 Author: Matteo Tomassetti

 Contact: www.matteotomassetti.com

 Based on Opinosis by Kavita Ganesan (http://kavita-ganesan.com/opinosis)

 Main difference with the Opinosis implementation:

 1) Before stitching the sentences to an anchor we use k-means to cluster their sentiment

 2) We use the Oxford Language Model API to evaluate whether a sentence is readable or not

'''

from opinosis import *

reviews = ["The battery is super cheap, but it's quite bulky to be honest.",\
           "The iPhone's battery is bulky but it's cheap.",\
           "The iPhone's battery is very cheap."]

# try with simple reviews
graph, nodes_PRI = get_graph(reviews)

# Model parameters

parameters = {
"SIGMA_VSN": 2,
"SIGMA_R": 2,
"GAP": 4,
"SIGMA_SIM": 0.3,
"MAX_SENTENCE_LENGTH": 15,
"NNEIGH": 5,
"OUT_CNN": 10,
"SENTIMENT_DIFF": 0.4
}

candidates = summarizer(graph,nodes_PRI,parameters=parameters)

tmp = remove_duplicates(candidates,parameters["SIGMA_SIM"])

# use the JP and R scores to discriminate which candidate sentences should be thrown out

for sentence, score in Counter(tmp).most_common():
	clean_sentence = untag(sentence)
	JP_score = joint_probability(clean_sentence)
	R_score = readability_score(clean_sentence)
	clean_sentence, score, JP_score, R_score



