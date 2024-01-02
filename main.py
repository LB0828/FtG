import os
import datetime
import argparse
from datetime import datetime
import warnings
import torch
from helper import get_num, read, read_name, read_file, get_ground_truth
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM, TrainingArguments, Trainer
import random
import numpy as np
from data import InstructionDataset
from peft import LoraConfig, get_peft_model


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

    # Step 4: Training
    print('LlaMA model loaded. Start training...' + '=' * 50)
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        warmup_steps=100,
        warmup_ratio=1e-3,
        num_train_epochs=30,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=50,
        optim="adamw_torch",
        save_total_limit=2,
        output_dir=configs.save_dir,
        evaluation_strategy="epoch",
        save_strategy="steps",
        save_steps=1000,
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
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset_path', type=str, default='/data/liuben/Prefix_T5/data/processed')
    parser.add_argument('-dataset', dest='dataset', default='WN18RR', help='Dataset to use, default: WN18RR')
    parser.add_argument('-model', default='T5Finetuner', help='Model Name')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-seed', dest='seed', default=42, type=int, help='Seed for randomization')
    parser.add_argument('-num_workers', type=int, default=64, help='Number of processes to construct batches')
    parser.add_argument('-save_dir', type=str, default='', help='')

    parser.add_argument('-pretrained_model', type=str, default='/data/liuben/llama2-7b', help='')
    parser.add_argument('-batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('-val_batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('-num_beams', default=40, type=int, help='Number of samples from beam search')
    parser.add_argument('-num_beam_groups', default=1, type=int, help='')
    parser.add_argument('-src_max_length', default=512, type=int, help='')
    parser.add_argument('-train_tgt_max_length', default=512, type=int, help='')
    parser.add_argument('-eval_tgt_max_length', default=30, type=int, help='')
    parser.add_argument('-epoch', dest='epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('-diversity_penalty', default=0., type=float, help='')

    parser.add_argument('-model_path', dest='model_path', default='', help='The path for reloading models')
    parser.add_argument('-optim', default='Adam', type=str, help='')
    parser.add_argument('-decoder', type=str, default='beam_search', help='[beam_search, do_sample, beam_sample_search, diverse_beam_search]')
    parser.add_argument('-log_text', action='store_true', help='')
    parser.add_argument('-use_prefix_search', action='store_true', help='')
    parser.add_argument('-src_descrip_max_length', default=0, type=int, help='')
    parser.add_argument('-tgt_descrip_max_length', default=0, type=int, help='')
    parser.add_argument('-use_soft_prompt', action='store_true', help='')
    parser.add_argument('-use_rel_prompt_emb', action='store_true', help='')
    parser.add_argument('-skip_n_val_epoch', default=0, type=int, help='')
    parser.add_argument('-seq_dropout', default=0., type=float, help='')
    parser.add_argument('-temporal', action='store_true', help='')
    parser.add_argument('-max_words', default=256, type=int, help='')

    configs = parser.parse_args()
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

