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

# open output file
pred_q = open("prediction.txt", "w")

# Load table-store
with open("data/table_store_v2.json", "rb") as f:
    ts_dataset = json.load(f)

# Load train and dev set
with open("data/train_set_v2.json", "rb") as f:
    eb_dataset_train = json.load(f)

eb_dataset = eb_dataset_train  # test dataset
f_retriever = BM25()  # relevance model

utils = Utils()
utils.init_explanation_bank_lemmatizer()

# fitting the models
corpus = []
original_corpus = []
question_train = []
ids = []
q_ids = []

for t_id, ts in tqdm(ts_dataset.items()):
    # facts lemmatization
    if "#" in ts["sentence_explanation"][-1]:
        fact = ts["sentence_explanation"][:-1]
    else:
        fact = ts["sentence_explanation"]
    lemmatized_fact = []
    original_corpus.append(fact)
    for chunck in fact:
        temp = []
        for word in nltk.word_tokenize(
            chunck.replace("?", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace(";", " ")
            .replace("-", " ")
        ):
            temp.append(utils.explanation_bank_lemmatize(word.lower()))
        if len(temp) > 0:
            lemmatized_fact.append(" ".join(temp))
    corpus.append(lemmatized_fact)
    ids.append(t_id)

for q_id, exp in tqdm(eb_dataset_train.items()):
    # concatenate question with candidate answer
    if exp["answerKey"] in exp["choices"]:
        question = (
            exp["question"]
            .replace("?", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace(";", " ")
            .replace("-", " ")
            .replace("'", "")
            .replace("`", "")
        )
        candidate = (
            exp["choices"][exp["answerKey"]]
            .replace("?", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace(";", " ")
            .replace("-", " ")
            .replace("'", "")
            .replace("`", "")
        )
    else:
        question = (
            exp["question"]
            .replace("?", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace(";", " ")
            .replace("-", " ")
            .replace("'", "")
            .replace("`", "")
        )
        candidate = ""
    question = question + " " + candidate
    temp = []
    # question lemmatization
    for word in nltk.word_tokenize(question):
        temp.append(utils.explanation_bank_lemmatize(word.lower()))
    lemmatized_question = " ".join(temp)
    question_train.append(lemmatized_question)
    q_ids.append(q_id)

f_retriever.fit(corpus, question_train, ids, q_ids)

exp_sim_overlaps = {}
overlap_flag = []
fact_reuse = {}
fact_similarity = {}

# compute the explanation ranking for each question
for q_id, exp in tqdm(eb_dataset.items()):
    # concatenate question with the answer
    if exp["answerKey"] in exp["choices"]:
        question = (
            exp["question"]
            .replace("?", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace(";", " ")
            .replace("-", " ")
            .replace("'", "")
            .replace("`", "")
        )
        candidate = (
            exp["choices"][exp["answerKey"]]
            .replace("?", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace(";", " ")
            .replace("-", " ")
            .replace("'", "")
            .replace("`", "")
        )
    else:
        question = (
            exp["question"]
            .replace("?", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace(";", " ")
            .replace("-", " ")
            .replace("'", "")
            .replace("`", "")
        )
        candidate = ""
    question = question + " " + candidate

    # lemmatization and stopwords removal
    temp = []
    for word in nltk.word_tokenize(question):
        if not word.lower() in stopwords.words("english"):
            temp.append(utils.explanation_bank_lemmatize(word.lower()))
    lemmatized_question = " ".join(temp)

    similar_questions = f_retriever.question_similarity([lemmatized_question])
    similar_facts = f_retriever.query([lemmatized_question])
    
    for res in similar_facts:
        if res["id"] in exp["explanation"]:
            if not exp["explanation"][res["id"]] == "CENTRAL":
                continue
            if not res["id"] in fact_reuse:
                    fact_reuse[res["id"]] = 0
                    fact_similarity[res["id"]] = []
            fact_reuse[res["id"]] += 1
            fact_similarity[res["id"]].append(res["score"])

    for res in similar_questions[:100]:
        #if res["id"]+"_"+q_id in overlap_flag:
           # continue
        overlap_flag.append(q_id+"_"+res["id"])
        if res["id"] != q_id:
            overlaps = 0
            count = 0
            for f_id in exp["explanation"]:
                if not exp["explanation"][f_id] == "CENTRAL":
                    continue
                count += 1
                if f_id in eb_dataset[res["id"]]["explanation"]:
                    overlaps += 1
            score = int(res["score"] * 10)
            if not score in exp_sim_overlaps.keys():
                exp_sim_overlaps[score] = []
            if overlaps > 0:
                exp_sim_overlaps[score].append(overlaps/count)
            else:
                exp_sim_overlaps[score].append(overlaps)

for overlap in exp_sim_overlaps:
    if len(exp_sim_overlaps[overlap]) > 0:
        print(overlap, sum(exp_sim_overlaps[overlap])/len(exp_sim_overlaps[overlap]))

fact_reuse_sim = {}

for fact in fact_reuse:
    if not fact_reuse[fact] in fact_reuse_sim:
        fact_reuse_sim[fact_reuse[fact]] = []
    fact_reuse_sim[fact_reuse[fact]].append(sum(fact_similarity[fact])/len(fact_similarity[fact]))

print("=======================================================================")
for reuse in fact_reuse_sim:
    print(reuse, sum(fact_reuse_sim[reuse])/len(fact_reuse_sim[reuse]))

pred_q.close()
