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
from models.FtGForCausalLM import FtGForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--lora_path', type=str, default=None)
parser.add_argument('--use_lora', action="store_true")
parser.add_argument('--llama', action="store_true")
parser.add_argument('--infer_file', type=str, required=True)
args = parser.parse_args()

max_new_tokens = 32
generation_config = dict(
    num_beams=10,
    num_return_sequences=10,
    num_beam_groups=1,
    max_new_tokens=max_new_tokens
)

infer_data = json.load(open(args.infer_file, 'r'))
instructions_list = []
labels = []
graph_id = []
test_graph_emb = torch.load("./test_graph_emb.pt")
for idx in range(len(infer_data)):
    text = infer_data[idx]['conversations'][0]['value']
    labels.append(infer_data[idx]['conversations'][1]['value'])
    instructions_list.append(
        {'instruction': f"Human: \n" + text + "\n\nAssistant: \n"}
    )
    graph_id.append(infer_data[idx]['graph_id'])


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

    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    model_config = (
        AutoConfig.from_pretrained(args.lora_path)
        if os.path.exists(os.path.join(args.lora_path, "config.json"))
        else None
    )

    if args.use_lora:
        model = FtGForCausalLM.from_pretrained(
            args.ckpt_path, torch_dtype=load_type, config=model_config
        )
        print('Loading additional FtG weights')
        if os.path.exists(os.path.join(args.lora_path, 'mm_projector.bin')):
            mm_projector_weights = torch.load(os.path.join(args.lora_path, 'mm_projector.bin'), map_location="cpu")
            mm_projector_weights = {k: v.to(load_type) for k, v in mm_projector_weights.items()}
        mm_projector_weights = {(k[11:] if k.startswith('base_model.') else k): v for k, v in mm_projector_weights.items()}
        if any(k.startswith('model.model.') for k in mm_projector_weights):
            mm_projector_weights = {(k[6:] if k.startswith('model.') else k): v for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)

        model = PeftModel.from_pretrained(model, args.lora_path)
        model.to(device)
        print('model is loaded')

    if device == torch.device('cpu'):
        model = model.float()
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
        graph = torch.FloatTensor(graph_id[idx]).to(device)
        graph_emb = test_graph_emb[graph_id[idx]].to(device)
        graph_emb = graph_emb.unsqueeze(0).unsqueeze(0).to(load_type)
        generation_output = model.generate(
            input_ids=inputs['input_ids'].to(device),
            graph=graph,
            graph_emb=graph_emb,
            **generation_config
        )
        generation_text = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
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
