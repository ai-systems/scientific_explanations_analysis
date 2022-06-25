import msgpack
import json
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

from unification_explanation.ranker.bm25 import BM25
from unification_explanation.ranker.relevance_score import RelevanceScore
from unification_explanation.ranker.tfidf import TFIDF
from unification_explanation.ranker.unification_score import UnificationScore
from unification_explanation.ranker.utils import Utils

# Load table-store
with open("data/cache/table_store.mpk", "rb") as f:
    ts_dataset = msgpack.unpackb(f.read(), raw=False)

# Load train and dev set
with open("data/cache/eb_train.mpk", "rb") as f:
    eb_dataset_train = msgpack.unpackb(f.read(), raw=False)


with open("data/cache/eb_dev.mpk", "rb") as f:
    eb_dataset_dev = msgpack.unpackb(f.read(), raw=False)

# Load table-store
with open("data/table_store_v2.json", "rb") as f:
    ts_dataset = json.load(f)

# Load train and dev set
with open("data/train_set_v2.json", "rb") as f:
    eb_dataset_train = json.load(f)

# open output file
pred_q = open("analysis_print.txt", "w")

utils = Utils()
utils.init_explanation_bank_lemmatizer()

knowledge_types = {}
edges = {}
edges_abstract = {}
edges_questions = {}
central_facts_count = {}
central_facts_questions = {}

for q_id, exp in tqdm(eb_dataset_train.items()):
    kt_flag = []
    edge_flag = []
    edge_abs_flag = []
    for fact in exp["explanation"]:
        overlap_fact = utils.explanation_bank_lemmatize(utils.clean_fact_for_overlaps(ts_dataset[fact]["explanation"]))
        clean_fact = utils.clean_fact(ts_dataset[fact]["explanation"])
        kt = ts_dataset[fact]["table_name"]
        if exp["explanation"][fact] == "CENTRAL":
            if not fact in central_facts_count:
                central_facts_count[fact] = 0
                central_facts_questions[fact] = []
            central_facts_count[fact] += 1
            central_facts_questions[fact].append(exp)
        if not kt in kt_flag:
            kt_flag.append(kt)
            if not kt in knowledge_types.keys():
                knowledge_types[kt] = 0
            knowledge_types[kt] += 1
        if exp["explanation"][fact] == "GROUNDING":
            for fact1 in exp["explanation"]:
                overlap_fact1 = utils.explanation_bank_lemmatize(utils.clean_fact_for_overlaps(ts_dataset[fact1]["explanation"]))
                clean_fact1 = utils.clean_fact(ts_dataset[fact1]["explanation"])
                kt1 = ts_dataset[fact1]["table_name"]
                if exp["explanation"][fact1] == "GROUNDING" and fact != fact1 and len(set(overlap_fact.split(" ")).intersection(set(overlap_fact1.split(" ")))) > 0:
                    if not kt+"_"+kt1 in edge_abs_flag:
                        edge_abs_flag.append(kt+"_"+kt1)
                        if not kt +"->"+ kt1 in edges_abstract.keys():
                            edges_abstract[kt+"->"+kt1] = 0
                        edges_abstract[kt+"->"+kt1] += 1
                    if not fact+"_"+fact1 in edge_flag:
                        edge_flag.append(fact+"_"+fact1)
                        if not clean_fact +"->"+ clean_fact1 in edges.keys():
                            edges[clean_fact+"->"+clean_fact1] = 0
                            edges_questions[clean_fact+"->"+clean_fact1] = []
                        edges[clean_fact+"->"+clean_fact1] += 1
                       # edges_questions[clean_fact+"->"+clean_fact1].append(exp["_question"]+" "+str(exp["_choices"][exp["_answerKey"]]))

for edge in sorted(edges_abstract, key=edges_abstract.get, reverse=True)[:20]:
    print(edge, edges_abstract[edge])

print("=================================================================================")

for edge in sorted(edges, key=edges.get, reverse=True)[:20]:
    print(edge, edges[edge])

print("=================================================================================")

for fact in sorted(central_facts_count, key=central_facts_count.get, reverse=True)[50:]:
    clean_fact = utils.clean_fact(ts_dataset[fact]["explanation"])
    print(clean_fact, central_facts_count[fact])
   # print("-----------------------------------------------------------------------------")
    #co_occurrence = {}
    #for exp in central_facts_questions[fact]:
        #print(exp["_question"]+" "+str(exp["_choices"][exp["_answerKey"]]))
        #for fact1 in exp["_explanation"]:
        #    if not fact1 in co_occurrence:
        #        co_occurrence[fact1] = 0
        #    co_occurrence[fact1] += 1
    #for fact1 in sorted(co_occurrence, key=co_occurrence.get, reverse=True)[:20]:
       # clean_fact1 = utils.clean_fact(ts_dataset[fact1]["_explanation"])
       # print(clean_fact1, co_occurrence[fact1])
pred_q.close()
