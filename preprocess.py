import os
from helper import read, read_name, read_file, get_configs
from transformers import LlamaTokenizer
import json
from tqdm import tqdm


PROMPT_DICT = {
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "instruction": "The head entity is {entity_name}. And the corresponding description is {entity_description}. Based on the description of the head entity, predict a plausible <mask> by fill in the given sentence: {sentence}."
}


def preprocess_data(configs):
    train_triples = read(configs, configs.dataset_path, configs.dataset, 'train2id.txt')
    valid_triples = read(configs, configs.dataset_path, configs.dataset, 'valid2id.txt')
    test_triples = read(configs, configs.dataset_path, configs.dataset, 'test2id.txt')
    all_triples = train_triples + valid_triples + test_triples
    original_ent_name_list, rel_name_list = read_name(configs, configs.dataset_path, configs.dataset)
    description_list = read_file(configs, configs.dataset_path, configs.dataset, 'entityid2description.txt', 'descrip')
    tokenizer = LlamaTokenizer.from_pretrained(configs.pretrained_model)
    src_description_list = tokenizer.batch_decode([descrip[1:] for descrip in tokenizer(description_list, max_length=configs.src_descrip_max_length, truncation=True).input_ids])
    ent_name_tokenized_list = tokenizer(original_ent_name_list, max_length=configs.train_tgt_max_length, truncation=True).input_ids
    ent_name_list = tokenizer.batch_decode([token[1:] for token in ent_name_tokenized_list])

    name_list_dict = {
        'train_triples': train_triples,
        'ent_name_list': ent_name_list,
        'rel_name_list': rel_name_list,
        'src_description_list': src_description_list,
    }
    train_json = []
    with tqdm(total=len(train_triples), desc='Constructing the training dataset...') as pbar:
        for idx, triple in enumerate(train_triples):
            head, tail, rel = triple
            head_name, tail_name, rel_name = ent_name_list[head], ent_name_list[tail], rel_name_list[rel]
            head_descrip, tail_descrip = src_description_list[head], src_description_list[tail]
            # Construct the pormpt json file
            sentence = head_name + ' ' + rel_name + ' ' + '<mask>'
            tgt = 'The <mask> is' + ' ' + tail_name
            instruction = PROMPT_DICT["instruction"].format(entity_name=head_name, entity_description=head_descrip, sentence=sentence)
            response = tgt
            data_idx = {
                'instruction': instruction,
                'response': response,
            }
            train_json.append(data_idx)
            pbar.update(1)
    train_tail_dataset = json.dumps(train_json)
    with open(os.path.join('./data_processed/WN18RR/train_tail_dataset.json'), 'w') as f:
        f.write(train_tail_dataset)


if __name__ == '__main__':
    configs = get_configs()
    preprocess_data(configs)
    



