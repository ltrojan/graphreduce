from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
import re
from textblob import TextBlob
import operator
import requests
import json
from sklearn.cluster import KMeans

#### MODEL parameters

parameters = {
"SIGMA_VSN": 15,
"SIGMA_R": 5,
"GAP": 4,
"SIGMA_SIM": 0.3,
"MAX_SENTENCE_LENGTH": 15,
"NNEIGH": 7,
"OUT_CNN": 10,
"SENTIMENT_DIFF": 0.2
}

####

def get_graph(reviews):
    tokenized_reviews = []
    for review in reviews:
        tokenized_reviews.append(" ".join([word.lower()+"/"+tag for word,tag in pos_tag(word_tokenize(review))]))
    
    filename = "tagged_reviews.tmp"
    
    with open(filename, "w") as f:
        for review in tokenized_reviews:
            f.write(review+"\n")
    f.close()
    #
    with open(filename,"r") as f:
        bigrams = []; nodes_PRI = defaultdict(list)
        for doc_id, line in enumerate(f):
            try:
                words = (line.strip()).split()
            except:
                continue
            #
            words = [word.replace("'s/VBZ","is/VBZ").replace("n't","not") for word in words]
            bigrams_in_document = list(zip(words, words[1:]))
            bigrams.extend(bigrams_in_document)
            # Positional Reference Information (PRI) for each node
            for j, word in enumerate(words):
                nodes_PRI[word].append([doc_id,j])
    #
    edges = Counter(bigrams)
    #
    with open("review_edges.tmp", "w") as f:
        for bigram in edges:
            f.write(" ".join([bigram[0], bigram[1]]) + 
                    " " + str(edges[bigram]))
            f.write("\n")
    #
    graph = nx.read_edgelist("review_edges.tmp", create_using=nx.DiGraph(),
                         data=(("count",int),))
    return graph, nodes_PRI

def is_valid_start(node,PRI,SIGMA_VSN):
    '''
        Check if a node is a Valid Starting Node (VSN)
        
        A node is VSN is the average PID <= SIGMA_VSN and
        
        its tag is one in the list above

        Also, if a node is common start word, we don't check the PID
    
    '''
    
    # check if a node is a common start word
    common_start_word = ["the","a","an","this","these","that","those",\
    					"when","if","for"]
    #
    if(node[:-2] in common_start_word):
    	return True
    # if not look at the average PDI
    PID = np.array(PRI)[:,1]
    valid_start_tags = ["JJ", "RB", "PRP$", "VBG", "NN", "DT"]
    if((np.mean(PID) <= SIGMA_VSN) and (node[-2:] in valid_start_tags)):
        return True
    else:
        return False

def is_valid_end(node):
    '''
        Check if a node is a Valid End Node (VEN)
        
        A node is a VEN is a period, a comma, or a coordinating conjunction
        
    '''
    
    if("/." in node or "/," in node or "/CC" in node):
        return True
    else:
        return False

def is_collapsible(graph,node,OUT_CNN):
    '''
        Check if a node is Collapsible
        
        A node is collapsible is it's a verb or if it has more than OUT_CNN outwards connections
    
    '''
    if(is_valid_end(node) is False):
	    if(len(graph[node])>OUT_CNN):
	    	return True
    
    if re.match(".*(/VB[A-Z]|/IN)", node) is not None:
        return True
    else:
        return False


def is_valid_path(sentence):
    '''
	    Determine if the sentence is a valid candidate.
    '''
    sentence = sentence.split()
    sent = " ".join(sentence)
    last = sentence[-1]
    w, t = last.split("/")
    if t in set(["TO", "VBZ", "IN", "CC", "WDT", "PRP", "DT", ","]):
        return False
    if re.match(".*(/DT)*.*(/VB)+.*(/PRP)+.*(/VB)+.*", sent): # This is it is
        return False
    if re.match(".*(very/RB very/RB).*", sent): # Very very very
        return False
    if re.match(".*(/JJ)*.*(/NN)+.*(/VB)+.*(/JJ)+.*", sent):
        return True
    elif re.match(".*(/RB)*.*(/JJ)+.*(/NN)+.*", sent) and not re.match(".*(/DT).*", sent):
        return True
    elif re.match(".*(/PRP|/DT)+.*(/VB)+.*(/RB|/JJ)+.*(/NN)+.*", sent):
        return True
    elif re.match(".*(/JJ)+.*(/TO)+.*(/VB).*", sent):
        return True
    elif re.match(".*(/RB)+.*(/IN)+.*(/NN)+.*", sent):
        return True
    else:
        return False

def PathScore(redundancy,length):
    '''
        Path Score
    '''
    return np.log2(length) * redundancy


def intersection(PRI, PRI_neigh, GAP):
    '''
        Compute the intersection between the PRIs of two nodes
        
        It returns the PRIs of the neighboring node that have the same SID
        
        and a different in PIDs less or equal than GAP
        
    '''
    
    PRI_intersect = []
    for the_pri in PRI:
        sid, pid = the_pri
        for the_pri_neigh in PRI_neigh:
            sid_n, pid_n = the_pri_neigh
            if sid == sid_n and abs(pid_n - pid) <= GAP:
                PRI_intersect.append(the_pri_neigh)
                break
    #
    return PRI_intersect

def rreplace(s, old, new, occurrence):
	li = s.rsplit(old, occurrence)
	return new.join(li)

# better way to stitch more sentences to an anchor
# 1) join the sentences ordered by score
# 2) sentences are connected based on their sentiment

def stitch(c_anchor, cc, SENTIMENT_DIFF):
	joined_sentence = " ".join(c_anchor)+" "
	#
	dummy = np.array(sorted([(key,score) for key, score in cc.items()],key=operator.itemgetter(1),reverse=True))
	all_ccs, scores = dummy[:,0], dummy[:,1].astype("float")
	#
	max_length = len(all_ccs)
	if(max_length==1):
		joined_sentence += all_ccs[0]
		return joined_sentence
	#
	sentiments = [TextBlob(untag(the_cc)).sentiment.polarity for the_cc in all_ccs]
	#
	group = KMeans(n_clusters=2)
	group.fit(np.array(sentiments).reshape(-1,1))
	groups = group.labels_
	groups_distance = abs(group.cluster_centers_[0]-group.cluster_centers_[1])
	#
	# if there are two groups of words with very different sentiment
	# then joined them together with `but/CC`
	if(groups_distance>=SENTIMENT_DIFF):
		# which group is more redundant
		redundancy = [np.mean(scores[groups==the_group]) for the_group in [0,1]]
		# put in front the group with the highest average redundancy
		sorted_groups = np.array(sorted(zip(redundancy,[0,1]),reverse=True))[:,1]
		group_0_stitched = " , ".join(np.array(sorted(zip(scores[groups==sorted_groups[0]],all_ccs[groups==sorted_groups[0]]),reverse=True))[:,1])
		group_0_stitched = rreplace(group_0_stitched," , "," and/DT ",1)
		joined_sentence += group_0_stitched
		#
		joined_sentence += " but/CC "
		#
		group_1_stitched = " , ".join(np.array(sorted(zip(scores[groups==sorted_groups[1]],all_ccs[groups==sorted_groups[1]]),reverse=True))[:,1])
		group_1_stitched = rreplace(group_1_stitched," , "," and/DT ",1)
		joined_sentence += group_1_stitched
	#
	# otherwise just stitch all the pieces with commas and a `and/DT` if more than three pieces
	else:
		stitched_sentences = " , ".join(all_ccs)
		joined_sentence += rreplace(stitched_sentences," , "," and/DT ",1)
	#
	return joined_sentence

def jaccard(sent1, sent2):
    '''
        Jaccard similarity score between two strings
    '''
    sent1 = set(sent1.split())
    sent2 = set(sent2.split())
    inter = sent1 & sent2
    return len(inter) / (len(sent1) + len(sent2) - len(inter))

#jaccard = np.vectorize(jaccard)

def remove_duplicates(tmp,SIGMA_SIM):
    '''
    
        tmp is a dictionary whose keys are sentences and values their scores.
        
        This functions prunes the keys who have high Jaccard similarity and only retains those with the highest score.
    
    '''
    sentences = np.array(list(tmp.keys()))
    
    if(len(sentences)==1):
        return tmp
    
    scores = np.array(list(tmp.values()))
    
    sentences_to_examine = list(sentences)
    sentences_to_keep = []
    i=0
    while(len(sentences_to_examine)>0):
        similar_sentences = np.array([jaccard(sentences_to_examine[i],dummy) >= SIGMA_SIM for dummy in sentences])
        already_kept = np.array([ss not in sentences_to_keep for ss in sentences])
        similar_sentences = similar_sentences & already_kept
        order = np.argsort(scores[similar_sentences])[::-1]
        sentences_to_keep.append(sentences[similar_sentences][order][0])
        # removing the other sentence
        for sentence_to_remove in np.unique(sentences[similar_sentences][order]):
            sentences_to_examine = list(filter(lambda x:x!=sentence_to_remove,sentences_to_examine))
    
    tmp_pruned = {key:tmp[key] for key in sentences_to_keep}
    return tmp_pruned

def untag(sentence):
    '''
        Removing the tags from a sentence and formatting it into a nice sentence
    '''
    
    untagged_sentence = " ".join([word.split("/")[0] for word in sentence.split()])
    untagged_sentence = untagged_sentence.capitalize()
    untagged_sentence = untagged_sentence.replace(" , ",", ")
    untagged_sentence = untagged_sentence.replace(" n't"," not")
    untagged_sentence = untagged_sentence.replace(" 's","'s")
    return untagged_sentence

def traverse(graph,nodes_PRI,cList,node,score,PRI,sentence,parent=[],parameters=parameters):
    '''
        Traversing the graph and building the candidate sentences
    
    '''
    # We only allow sentences that are at most MAX_SENTENCE_LENGTH long
    if(len(sentence)>parameters["MAX_SENTENCE_LENGTH"]):
        return
    #
    redundancy = len(PRI)
    #
    # traverse the graph only if a node meets the rudandancy criterion
    if(redundancy >= parameters["SIGMA_R"] or is_valid_end(node)):
        if(is_valid_end(node)):
            # remove the /CC or the punctation at the end of the sentence if present
            del sentence[-1]
            # check if the sentence is valid
            if(is_valid_path(" ".join(parent + sentence))):
                final_score = score/len(parent + sentence)
                cList[" ".join(sentence)] = score
            return
        #
        # if not keep traversing the graph
        neighbors = graph[node]
        for neighbor in neighbors:
            if (neighbor in (sentence + parent) and neighbor[-2:] not in ["DT"]):
            	continue
            new_sentence = sentence[:]
            new_sentence.append(neighbor)
            PRI_new = intersection(PRI,nodes_PRI[neighbor],parameters["GAP"])
            redundancy = len(PRI_new)
            new_score = score + PathScore(redundancy,len(new_sentence))
            if(is_collapsible(graph,neighbor,parameters["OUT_CNN"]) and not parent):
                c_anchor = new_sentence
                tmp = defaultdict(int)
                best_neighbors = np.array(sorted([(nn,len(nodes_PRI[nn])) for nn in graph[neighbor]],key=operator.itemgetter(1),reverse=True))[:,0][:parameters["NNEIGH"]]
                for neighbor_an in best_neighbors:#graph[neighbor]:
                    traverse(graph,nodes_PRI,tmp,neighbor_an,0,PRI_new,[neighbor_an], new_sentence,parameters=parameters)
                    if("" in tmp): del tmp[""]
                    if(tmp):
                        tmp = remove_duplicates(tmp,parameters["SIGMA_SIM"])
                        CCPath_score = np.mean(list(tmp.values()))
                        final_score = new_score + CCPath_score
                        if(np.isnan(final_score)):
                            print("NAN=",new_score, CCPath_score,tmp)
                        stitched_sent = stitch(c_anchor, tmp, parameters["SENTIMENT_DIFF"])
                        if(is_valid_path(stitched_sent)):
                            cList[stitched_sent] = final_score
            else:
                traverse(graph,nodes_PRI,cList,neighbor,new_score,PRI_new,new_sentence,parent,parameters=parameters)

def summarizer(graph,nodes_PRI,parameters=parameters):
	# summarization algorithm
	candidates = defaultdict(int)
	for node, PRI in nodes_PRI.items():
	    if(is_valid_start(node,PRI,parameters["SIGMA_VSN"])):
	        PRI = nodes_PRI[node]
	        score = 0
	        cList = defaultdict(int)
	        # sentence formed so far
	        sentence = [node]
	        # recursively traverse the graph and create candidate sentences
	        traverse(graph,nodes_PRI,cList,node,score,PRI,sentence,[],parameters=parameters)
	        candidates.update(cList)
	return candidates

### CHECK IF THE REVIEWS MAKE SENSE
### Using the Microsoft Oxford Language Model API

# ---> Please use your key <---

OXFORD_LANGUAGE_MODEL_ENDPOINT = "https://api.projectoxford.ai/text/weblm/v1.0/"
SUBSCRIPTION_KEY_NAME = ""
DEFAULT_SUBSCRIPTION_KEY = ""

def conditional_probability(query):
    '''
    
    Calculate the Conditional Probability that a particular word will follow a given sequence of words

    Examples:

    query = {"words": "this is", "word": "sparta"}
        
    '''
    data = json.dumps({"queries" : [query]})
    headers = {SUBSCRIPTION_KEY_NAME: DEFAULT_SUBSCRIPTION_KEY,
                "Content-Type": "application/json",
                "Content-Length": str(len(data))}
    #
    params = {
             "model" : "body",
             "order" : 1
            }
    #
    r = requests.post(OXFORD_LANGUAGE_MODEL_ENDPOINT + "calculateConditionalProbability", headers=headers,  params=params, data=data)
    #
    if r.status_code == 200:
        return 10.**(r.json()["results"][0]["probability"])
    else:
        return 10.**-30.

def readability_score(input_string, n_grams_order = 3):
    '''
    
    Readability score from Eq. 4 in Ganesan et al. 2012
    
    
    S_red (w_1...w_n) = 1/K * PROD_q^n P(w_k,w_{k-q+1}...w_{k-1})
    
    '''
    
    input_string = input_string.lower()
    words = input_string.split()
    
    k = range(n_grams_order-1,len(words))
    
    probability = 1
    for kk in k:
        partial_probability = conditional_probability({"word": words[kk],"words": words[kk-n_grams_order+1] + " " + words[kk-n_grams_order+2]})
        probability *= partial_probability
    
    return np.log2(probability) / len(k)

def joint_probability(query):
    '''
    
    Calculate the Conditional Probability that a particular word will follow a given sequence of words

    Examples:

    query = "this is sparta"
        
    '''
    data = json.dumps({"queries" : [query]})
    headers = {SUBSCRIPTION_KEY_NAME: DEFAULT_SUBSCRIPTION_KEY,
                "Content-Type": "application/json",
                "Content-Length": str(len(data))}
    #
    params = {
             "model" : "body",
             "order" : 1
            }
    #
    r = requests.post(OXFORD_LANGUAGE_MODEL_ENDPOINT + "calculateJointProbability", headers=headers,  params=params, data=data)
    #
    if r.status_code == 200:
        return r.json()["results"][0]["probability"]
    else:
        return -99




