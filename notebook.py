# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

directory = "./my_model_save/distilbert-base-nli-stsb-mean-tokens"

embedder = SentenceTransformer(directory)

########################
import re

#regular expressions
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])" ###

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text) ###
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
########################



dump = """A group of kids is playing in a yard and an old man is standing in the background.
A group of children is playing in the house and there is no man standing in the background.
The young boys are playing outdoors and the man is smiling nearby.
The kids are playing outdoors near a man with a smile.
Two dogs are fighting.
A brown dog is attacking another animal in front of the man in pants.
A person is riding the bicycle on one wheel.
A person on a black motorbike is doing tricks with a jacket.
A man with a jersey is dunking the ball at a basketball game.
Two young women are sparring in a kickboxing fight.
Two young women are not sparring in a kickboxing fight.
Two women are sparring in a kickboxing match.
Two children are lying in the snow and are making snow angels.
People wearing costumes are gathering in a forest and are looking in the same direction.
People are looking at some costumes gathered in the vicinity of the forest.
A little girl is looking at a woman in costume.
A kid swimming in the ocean is tossing a coin into the pool, near the ma.n
A father is launching the daughter in a swimming pool.
The man is tossing a kid into the swimming pool that is near the ocean.
Four girls are doing backbends and playing in the garden.
Two groups of people are playing football.
Two teams are competing in a baseball game.
Five wooden stands are in front of each child's hut.
The young boy is climbing the wall made of rock.
"""


corpus = split_into_sentences(dump)

corpus_embeddings = embedder.encode(corpus)

# Perform kmean clustering
#num_clusters = 5
#clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
clustering_model = AgglomerativeClustering(distance_threshold=20, 
                                           compute_full_tree=True, 
                                           n_clusters=None)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_
clustered_sentences = [[] for i in range(max(cluster_assignment)+1)]

for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print(cluster)
    print("")

