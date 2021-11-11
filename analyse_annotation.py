import json

# convert JSONL into JSON and extract the full annotation
with open('annotation.jsonl', 'r') as json_file:
    json_list = list(json_file)

full_annotation = []
for json_str in json_list:
    full_annotation.append(json.loads(json_str))


transition_probabilities = {}

for question in full_annotation:
    # for each question, use the position of each label as an index to order the sequence
    labels = question["label"]
    labels_index = {}
    for label in labels:
        labels_index[label[0]] = label[2]
    ordered_indexes = sorted(list(labels_index.keys()))

    #count frequencies of sequences strarting from the question
    previous_label = "Question"
    for index in ordered_indexes:
        current_label = labels_index[index]
        if not previous_label in transition_probabilities:
            transition_probabilities[previous_label] = {}
        if not current_label in transition_probabilities[previous_label]:
            transition_probabilities[previous_label][current_label] = 0
        transition_probabilities[previous_label][current_label] += 1
        previous_label = current_label

# normalize transition probabilities
for origin in transition_probabilities:
    total = 0
    for target in transition_probabilities[origin]:
        total += transition_probabilities[origin][target]
    for target in transition_probabilities[origin]:
        transition_probabilities[origin][target] /= total

#print transitions
for origin in transition_probabilities:
    print(origin)
    for target in transition_probabilities[origin]:
        print("\t-", target, transition_probabilities[origin][target])   


