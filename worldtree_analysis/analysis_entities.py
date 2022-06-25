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

utils = Utils()
utils.init_explanation_bank_lemmatizer()

# entities dict
entities_freq = {}
entities_depth = {}

for q_id, exp in tqdm(eb_dataset.items()):
    flag = []
    for fact in exp["explanation"]:
        if exp["explanation"][fact] == "CENTRAL": #or exp["explanation"][fact] == "GROUNDING":
            entities = utils.recognize_entities(utils.clean_fact(ts_dataset[fact]["explanation"]))
            for entity in entities:
                if entity[0] in flag:
                    break
                flag.append(entity[0])
                if not entity[0] in entities_freq:
                    entities_freq[entity[0]] = 0
                    entities_depth[entity[0]] = entity[1]
                entities_freq[entity[0]] += 1

for entity in  sorted(entities_freq, key=entities_freq.get, reverse=True)[:50]:
    print(entity, entities_freq[entity], entities_depth[entity])

pred_q.close()
