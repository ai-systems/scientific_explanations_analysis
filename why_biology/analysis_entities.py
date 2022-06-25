from unification_explanation.ranker.utils import Utils
import json
from tqdm import tqdm

# convert JSONL into JSON and extract the full annotation
with open('annotation.jsonl', 'r') as json_file:
        json_list = list(json_file)

utils = Utils()
utils.init_explanation_bank_lemmatizer()

full_annotation = []
for json_str in json_list:
    full_annotation.append(json.loads(json_str))

entities_freq = {}
for question in tqdm(full_annotation):
    flag_found = []
    data = question["data"].split("\n\n")
    for paragraph in data:
        entities = utils.recognize_entities(paragraph)
        for entity in entities:
            if entity[0] in flag_found:
                break
            flag_found.append(entity[0])
            if not entity[0] in entities_freq:
                entities_freq[entity[0]] = 0
            entities_freq[entity[0]] += 1

for entity in sorted(entities_freq, key=entities_freq.get, reverse=True)[:100]:
    print(entity, entities_freq[entity])

