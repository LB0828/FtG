import os
import datetime
import argparse
from datetime import datetime
import warnings
import torch
from helper import get_num, read, read_name, read_file, get_ground_truth, get_configs
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM, TrainingArguments, Trainer
import random
import numpy as np
from data import InstructionDataset, InstructionEvalDataset
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel


def make_supervised_data_module(configs, tokenizer, train_triples, valid_triples, name_list_dict):
    dataset_cls = (InstructionDataset)
    train_dataset = dataset_cls(configs, tokenizer, train_triples, name_list_dict)
    eval_dataset = dataset_cls(configs, tokenizer, valid_triples, name_list_dict)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def main():
    # Step 1: Load data and tokenize the data
    train_triples = read(configs, configs.dataset_path, configs.dataset, 'train2id.txt')
    valid_triples = read(configs, configs.dataset_path, configs.dataset, 'valid2id.txt')
    test_triples = read(configs, configs.dataset_path, configs.dataset, 'test2id.txt')
    all_triples = train_triples + valid_triples + test_triples

    original_ent_name_list, rel_name_list = read_name(configs, configs.dataset_path, configs.dataset)
    description_list = read_file(configs, configs.dataset_path, configs.dataset, 'entityid2description.txt', 'descrip')
    tokenizer = LlamaTokenizer.from_pretrained(configs.pretrained_model)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"
    print('Tokenizer loaded. Tokenizing the data...' + '=' * 50)
    src_description_list = tokenizer.batch_decode([descrip[1:] for descrip in tokenizer(description_list, max_length=configs.src_descrip_max_length, truncation=True).input_ids])
    tgt_description_list = tokenizer.batch_decode([descrip[1:] for descrip in tokenizer(description_list, max_length=configs.tgt_descrip_max_length, truncation=True).input_ids])
    # TODO: for LlaMa2-chat specialized pormpt format
    ent_name_tokenized_list = tokenizer(original_ent_name_list, max_length=configs.train_tgt_max_length, truncation=True).input_ids
    ent_name_list = tokenizer.batch_decode([token[1:] for token in ent_name_tokenized_list])
    if configs.tgt_descrip_max_length > 0:
        ent_name_tokenized_list_with_descrip = tokenizer([ent_name + '[' + tgt_description_list[i] + ']' for i, ent_name in enumerate(original_ent_name_list)], 
                                                         max_length=configs.train_tgt_max_length,
                                                         truncation=True).input_ids
    name_list_dict = {
        'original_ent_name_list': original_ent_name_list,
        'ent_name_list': ent_name_list,
        'rel_name_list': rel_name_list,
        'src_description_list': src_description_list,
        'ent_name_tokenized_list': ent_name_tokenized_list,
        'ent_name_tokenized_list_with_descrip': ent_name_tokenized_list_with_descrip,
    }
    print('Data tokenized. Loading ground truth...' + '=' * 50)
    train_tail_ground_truth, train_head_ground_truth = get_ground_truth(configs, train_triples)
    all_tail_ground_truth, all_head_ground_truth = get_ground_truth(configs, all_triples)
    ground_truth_dict = {
        'train_tail_ground_truth': train_tail_ground_truth,
        'train_head_ground_truth': train_head_ground_truth,
        'all_tail_ground_truth': all_tail_ground_truth,
        'all_head_ground_truth': all_head_ground_truth,
    }

    # Step 2: Construct training dataloaders
    print('Ground truth loaded. Constructing training dataloaders...' + '=' * 50)
    datamodule = make_supervised_data_module(configs, tokenizer, train_triples, valid_triples, name_list_dict)

    # Step 3: Loading the LlaMA model
    print('Training dataloaders constructed. Loading the LlaMA model...' + '=' * 50)
    if configs.model_path != '':
        model = LlamaForCausalLM.from_pretrained(configs.pretrained_model, torch_dtype=torch.float16).to("cuda:0")
        model = PeftModel.from_pretrained(model, configs.model_path, torch_dtype=torch.float16).to("cuda:0")
        model = model.eval()
    else:
        model = LlamaForCausalLM.from_pretrained(configs.pretrained_model, torch_dtype=torch.float16, use_cache=False, device_map='auto')
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Step 4: Training&Inference
    if configs.model_path != '':
        answer = []
        predict = []
        print('LlaMA model loaded. Start inference...' + '=' * 50)
        with tqdm(total=len(valid_triples), desc="Inference Progress") as pbar:
            for sample in InstructionEvalDataset(configs, tokenizer, valid_triples, name_list_dict):
                input_ids = sample['input_ids'].unsqueeze_(0).to("cuda:0")
                label = sample['label']
                generate_ids = model.generate(input_ids=input_ids, max_new_tokens=20)
                result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
                final_result = result[0].split('\n')[-1]
                answer.append(label)
                predict.append(final_result)
                pbar.update(1)
        acc = 0
        for idx in range(len(answer)):
            with open(os.path.join(configs.save_dir, 'answer.txt'), 'a') as f:
                f.write(predict[idx] + '\t' + answer[idx] + '\n')
            if answer[idx] in predict[idx] or predict[idx] in answer[idx]:
                acc += 1
        print('acc: ', acc / len(answer))
    else:
        print('LlaMA model loaded. Start training...' + '=' * 50)
        training_args = TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            warmup_steps=200,
            warmup_ratio=1e-3,
            num_train_epochs=12,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=50,
            optim="adamw_torch",
            save_total_limit=2,
            load_best_model_at_end=True,
            output_dir=configs.save_dir,
            evaluation_strategy="epoch",
            save_strategy="steps",
            save_steps=2000,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
        )
        trainer = Trainer(
            model=model, tokenizer=tokenizer, args=training_args, **datamodule
        )
        model.config.use_cache = False
        trainer.train()
        model.save_pretrained(configs.save_dir)

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    configs = get_configs()
    n_ent = get_num(configs.dataset_path, configs.dataset, 'entity')
    n_rel = get_num(configs.dataset_path, configs.dataset, 'relation')
    configs.n_ent = n_ent
    configs.n_rel = n_rel
    configs.vocab_size = LlamaConfig.from_pretrained(configs.pretrained_model).vocab_size
    configs.model_dim = LlamaConfig.from_pretrained(configs.pretrained_model).hidden_size
    if configs.save_dir == '':
        configs.save_dir = os.path.join('./checkpoint_lora', configs.dataset + '-' + str(datetime.now()))
    os.makedirs(configs.save_dir, exist_ok=True)
    seed_everything(configs.seed)
    print(configs)
    main()

