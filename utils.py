import os
import json


def read(dataset_path, dataset, filename):
    file_name = os.path.join(dataset_path, dataset, 'transformer_data', filename)
    with open(file_name) as file:
        lines = file.read().strip().split('\n')
    triples = []
    for line in lines:
        split = line.split('\t')
        triples.append(split)
    return triples


def read_dict(dataset_path, dataset):
    ent2id_dict = {}
    rel2id_dict = {}
    for split in ["entities.dict", "relations.dict"]:
        file_name = os.path.join(dataset_path, dataset, 'transformer_data', split)
        with open(file_name, 'r') as file:
            lines = file.read().strip().split('\n')
        dict = {}
        for line in lines:
            datapoint = line.split('\t')
            dict[datapoint[1]] = int(datapoint[0])
        if split == "entities.dict":
            ent2id_dict = dict
        else:
            rel2id_dict = dict
    return ent2id_dict, rel2id_dict


def read_file(dataset_path, dataset, filename, mode='descrip'):
    id2name = []
    file_name = os.path.join(dataset_path, dataset, filename)
    with open(file_name, encoding='utf-8') as file:
        lines = file.read().strip('\n').split('\n')
    for i in range(1, len(lines)):
        ids, name = lines[i].split('\t')
        if mode == 'descrip':
            name = name.split(' ')
            name = ' '.join(name)
        id2name.append(name)
    return id2name


def read_name(dataset_path, dataset):
    ent_name_file = 'entityid2name.txt'
    rel_name_file = 'relationid2name.txt'
    ent_name_list = read_file(dataset_path, dataset, ent_name_file, 'name')
    rel_name_list = read_file(dataset_path, dataset, rel_name_file, 'name')
    return ent_name_list, rel_name_list



'''
orig_data: {"instruction": "xxxxxxxxx", "input": "", "output": "xxxxxxxxxx"}
convert: {
    "id": xxx,
    "conversations":[
        {"from": "human", "value": "xxxxxxxxx"},
        {"from": "assistant", "value": "xxxxxxxxxx"},
    ]
}
{"prompt": "Please answer the following commonsense question. Please first explain each candidate answer, then select only one answer that is most relevant to the question and provide reasons.\nQuestion: What do people typically do while playing guitar?\nCandidate Answers: cry, hear sounds, singing, arthritis, making music.\nAnswer: Let's think step by step.\n", "answer": "singing", "candidates": ["cry", "hear sounds", "singing", "arthritis", "making music"]}
'''
def convert_file_to_conv(dataset_path, dataset, filename):
    instructions_list = json.load(open(os.path.join(dataset_path, dataset, 'unshuffle', filename), 'r'))
    conversations = []
    for idx, instructions in enumerate(instructions_list):
        conversation = []
        prompt = f"Please answer the following commonsense question and select only one answer from the candidates that is most relevant to the question.\nQuestion: {instructions['prompt']}\nCandidate Answers: {', '.join(instructions['candidates'])}."
        conversation.append({"from": "human", "value": prompt})
        conversation.append({"from": "assistant", "value": instructions['output']})
        conversations.append({
            "id": str(idx),
            "conversations": conversation
        })
    with open(os.path.join(dataset_path, dataset, 'unshuffle', 'converted_' + filename), 'w') as f:
        json.dump(conversations, f)


# def adding_context(dataset_path, dataset, filename):


if __name__ == "__main__":
    convert_file_to_conv("/data", "FB15k-237N", "test.json")