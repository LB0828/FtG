import torch
from tqdm import tqdm
from transformers import (
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
from peft import PeftModel
import argparse
import json
import os
import random
import numpy as np
from graph_llm import GraphLLM


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--kge_model', type=str, required=True)
parser.add_argument('--projected_emb_path', type=str)
parser.add_argument('--kge_model_path', type=str, required=True)
parser.add_argument('--lora_path', type=str, default=None)
parser.add_argument('--use_lora', action="store_true")
parser.add_argument('--llama', action="store_true")
parser.add_argument('--infer_file', type=str, required=True)
args = parser.parse_args()

max_new_tokens = 32
# generation_config = dict(
#     temperature=0.9,
#     top_k=30,
#     top_p=0.6,
#     do_sample=True,
#     num_beams=1,
#     repetition_penalty=1.2,
#     max_new_tokens=max_new_tokens,
# )
generation_config = dict(
    num_beams=10,
    num_return_sequences=10,
    num_beam_groups=1,
    max_new_tokens=max_new_tokens
)

infer_data = json.load(open(args.infer_file, 'r'))
instructions_list = []
labels = []
graph_tokens_list = []
for idx in range(len(infer_data)):
    graph_tokens_list.append(infer_data[idx]['graph_tokens'])
    text = infer_data[idx]['conversations'][0]['value']
    labels.append(infer_data[idx]['conversations'][1]['value'])
    instructions_list.append(
        {'instruction': f"Human: \n" + text + "\n\nAssistant: \n"}
    )


if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.llama:
        tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    projected_emb = torch.load(args.projected_emb_path)
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    model_config = (
        AutoConfig.from_pretrained(args.ckpt_path)
        if os.path.exists(os.path.join(args.ckpt_path, "config.json"))
        else None
    )

    if args.use_lora:
        base_model = AutoModelForCausalLM.from_pretrained(
            "/llama2-7b", torch_dtype=load_type, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, args.lora_path, torch_dtype=load_type)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path, torch_dtype=load_type, device_map="auto", config=model_config
        )
    if device == torch.device('cpu'):
        model = model.float()
    model = GraphLLM(model, args.kge_model, args.kge_model_path)
    model.eval()
    print("Load model successfully!")

    acc = 0
    ranks = []
    for idx, instruction in tqdm(enumerate(instructions_list), total=len(instructions_list)):
        inputs = tokenizer(
            instruction['instruction'],
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids=inputs['input_ids'].to(device)
        input_embs = model.llama_model.model.model.embed_tokens(input_ids)
        graph_tokens = torch.tensor(graph_tokens_list[idx], dtype=torch.long).to(device)
        prefix_embs = projected_emb[graph_tokens].to(device)
        input_embs = torch.cat((prefix_embs, input_embs), dim=1)

        generation_output = model.llama_model.generate(
            inputs_embeds=input_embs,
            **generation_config
        )
        generation_text = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        # if labels[idx] in generation_text.split("Assistant:")[1]:
        #     acc += 1
        generation_text = [text.split("Assistant:")[1].strip() for text in generation_text]
        if labels[idx] in generation_text:
            top_entities = set()
            rank = 1
            for text in generation_text:
                if labels[idx] in text:
                    ranks.append(rank)
                    break
                if text not in top_entities:
                    top_entities.add(text)
                    rank += 1
        else:
            ranks.append(random.randint(11, 14541))
    print(args.ckpt_path)
    print(args.infer_file)
    ranks = np.array(ranks, dtype=np.float32)
    mrr = (1. / ranks).mean(axis=0)
    hits1 = np.sum(ranks == 1, axis=0) / len(ranks)
    hits3 = np.sum(ranks <=3, axis=0) / len(ranks)
    hits10 = np.sum(ranks <= 10, axis=0) / len(ranks)
    print('mrr:', str(mrr) + '\n', 'hits1:', str(hits1) + '\n', 'hits3:', str(hits3) + '\n', 'hits10:', str(hits10))

    
