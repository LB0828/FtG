from torch.utils.data import Dataset
import torch
import copy


PROMPT_DICT = {
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
}

class InstructionDataset(Dataset):
    def __init__(self, configs, tokenizer, triples, name_list_dict):
        self.configs = configs
        self.tokenizer = tokenizer
        self.triples = triples
        self.original_ent_name_list = name_list_dict['original_ent_name_list']
        self.ent_name_list = name_list_dict['ent_name_list']
        self.rel_name_list = name_list_dict['rel_name_list']
        self.src_description_list = name_list_dict['src_description_list']
        self.ent_name_tokenized_list = name_list_dict['ent_name_tokenized_list']
        self.ent_name_tokenized_list_with_descrip = name_list_dict['ent_name_tokenized_list_with_descrip']
        self.max_length = self.configs.max_words

    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, index):
        triple = self.triples[index]
        head, tail, rel = triple
        head_name, tail_name, rel_name = self.original_ent_name_list[head], self.original_ent_name_list[tail], self.rel_name_list[rel]
        if self.configs.src_descrip_max_length > 0:
            head_descrip = '[' + self.src_description_list[head] + ']'
            tail_descrip = '[' + self.src_description_list[tail] + ']'
        else:
            head_descrip, tail_descrip = '', ''
        # Constrcut the input sequence and corresponding labels
        src = head_name + ' ' + head_descrip + ' | ' + rel_name + ' | '
        tgt = tail_name + ' ' + tail_descrip
        src_tokenized = self.tokenizer(src, max_length=self.max_length, truncation=True)
        tgt_tokenized = self.tokenizer(tgt, max_length=self.max_length, truncation=True)
        prompt = PROMPT_DICT["prompt_no_input"].format(instruction=src)
        example = prompt + tgt
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)

        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_length - example.shape[0]
        if padding > 0:
            example = torch.cat(
                (torch.zeros(padding, dtype=torch.int64) - 1, example), dim=0
            )
        elif padding < 0:
            example = example[: self.max_length]
        labels = copy.deepcopy(example)
        labels[padding:(padding+len(prompt))] = -1
        example_mask = example.ge(0)
        labels_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~labels_mask] = -100
        example_mask = example_mask.float()
        labels_mask = labels_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }
