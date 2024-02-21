import os
import torch
from utils import read, read_name, read_dict
from tqdm import tqdm
import json
from model import KGEModel
from collections import defaultdict
from torch.utils.data import DataLoader
from dataloader import TestDataset
import random


def get_ground_truth(triples):
    tail_ground_truth, head_ground_truth = defaultdict(list), defaultdict(list)
    for triple in triples:
        head, relation, tail = triple
        tail_ground_truth[(head, relation)].append(tail)
        head_ground_truth[(tail, relation)].append(head)
    return tail_ground_truth, head_ground_truth


def convert_triples_to_id(triples, ent2id_dict, rel2id_dict):
    triples2id = []
    for triple in triples:
        head, relation, tail = triple
        head_id = ent2id_dict[head]
        relation_id = rel2id_dict[relation]
        tail_id = ent2id_dict[tail]
        triples2id.append([head_id, relation_id, tail_id])
    return triples2id


def get_entity_related_context(dataset_path, dataset):
    train_triples_raw = read(dataset_path, dataset, 'train.txt')
    valid_triples_raw = read(dataset_path, dataset, 'valid.txt')
    test_triples_raw = read(dataset_path, dataset, 'test.txt')
    ent2id_dict, rel2id_dict = read_dict(dataset_path, dataset)
    train_triples = convert_triples_to_id(train_triples_raw, ent2id_dict, rel2id_dict)
    valid_triples = convert_triples_to_id(valid_triples_raw, ent2id_dict, rel2id_dict)
    test_triples = convert_triples_to_id(test_triples_raw, ent2id_dict, rel2id_dict)
    ent_name_list, rel_name_list = read_name(dataset_path, dataset)
    rel_name_list = [name.replace(' , ', '/') for name in rel_name_list]

    return train_triples, valid_triples, test_triples, ent2id_dict, rel2id_dict, ent_name_list, rel_name_list


def get_entity_candidate(kge_model_path, dataset_path, dataset):
    # load the entity & rel embedding from the kge model
    with open(os.path.join(kge_model_path, 'config.json')) as f:
        kge_args = json.load(f)

    kge_model = KGEModel(
        model_name=kge_args['model'],
        nentity=kge_args['nentity'],
        nrelation=kge_args['nrelation'],
        hidden_dim=kge_args['hidden_dim'],
        gamma=kge_args['gamma'],
        double_entity_embedding=kge_args['double_entity_embedding'],
        double_relation_embedding=kge_args['double_relation_embedding'],
    ).cuda()
    checkpoint = torch.load(os.path.join(kge_model_path, 'checkpoint'))
    kge_model.load_state_dict(checkpoint['model_state_dict'])
    kge_model.entity_embedding.requires_grad = False
    kge_model.relation_embedding.requires_grad = False
    train_triples, valid_triples, test_triples, ent2id_dict, rel2id_dict, ent_name_list, rel_name_list = get_entity_related_context(dataset_path, dataset)
    all_triples = train_triples + valid_triples + test_triples
    all_triples = [tuple(triple) for triple in all_triples]
    kge_model.eval()
    metrics = []
    tail_ground_truth, head_ground_truth = get_ground_truth(all_triples)
    
    with torch.no_grad():
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_triples,
                kge_args['nentity'],
                kge_args['nrelation'],
                'head-batch'
            ),
            batch_size=1,
            num_workers=max(1, kge_args['cpu_num']),
            collate_fn=TestDataset.collate_fn
        )
        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_triples,
                kge_args['nentity'],
                kge_args['nrelation'],
                'tail-batch'
            ),
            batch_size=1,
            num_workers=max(1, kge_args['cpu_num']),
            collate_fn=TestDataset.collate_fn
        )
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        head_instruction_samples = []
        tail_instruction_samples = []
        instruction_samples = []
        with torch.no_grad():
            for test_dataset in test_dataset_list:
                with tqdm(total=len(test_triples)) as pbar:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        head, rel, tail = positive_sample.squeeze().tolist()
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()
                        batch_size = positive_sample.size(0)
                        score = kge_model((positive_sample, negative_sample), mode)
                        score += filter_bias
                        argsort = torch.argsort(score, dim=1, descending=True)
                        if mode == 'head-batch': 
                            positive_arg = positive_sample[:, 0]
                            prompt = "What/Who/When/Where/Why" + " " + rel_name_list[rel] + " " + ent_name_list[tail] + "?"
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                            prompt = ent_name_list[head] + " " + rel_name_list[rel] + "?"
                        else:
                            raise ValueError('mode %s not supported' % mode)
                        for i in range(batch_size):
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            # get top 100 entities
                            candidates = argsort[i, :30].detach().tolist()
                            candidates = [ent_name_list[candidate] for candidate in candidates]
                            assert ranking.size(0) == 1
                            ranking = 1 + ranking.item()

                            instruction_samples.append(
                                {
                                    "prompt": prompt,
                                    "candidates": candidates,
                                    "ranking": ranking,
                                    "output": ent_name_list[positive_arg[i].item()]
                                }
                            )
                            if mode == 'head-batch':
                                head_instruction_samples.append(
                                    {
                                        "prompt": prompt,
                                        "candidates": candidates,
                                        "ranking": ranking,
                                        "output": ent_name_list[positive_arg[i].item()]
                                    }
                                )
                            else:
                                tail_instruction_samples.append(
                                    {
                                        "prompt": prompt,
                                        "candidates": candidates,
                                        "ranking": ranking,
                                        "output": ent_name_list[positive_arg[i].item()]
                                    }
                                )
                            metrics.append(1.0 / ranking)
                        pbar.update(batch_size)
        print(len(metrics))
        print('*'*30)
        print("MRR: {}".format(sum(metrics) / len(metrics)))
        assert len(instruction_samples) == len(test_triples) * 2
        with open(os.path.join(dataset_path, dataset, 'unshuffle_top30', 'test_instruction_samples_top_30_unpop.json'), 'w') as f:
            f.write(json.dumps(instruction_samples))
        with open(os.path.join(dataset_path, dataset, 'unshuffle_top30', 'test_head_instruction_samples_top_30_unpop.json'), 'w') as f1:
            f1.write(json.dumps(head_instruction_samples))
        with open(os.path.join(dataset_path, dataset, 'unshuffle_top30', 'test_tail_instruction_samples_top_30_unpop.json'), 'w') as f2:
            f2.write(json.dumps(tail_instruction_samples))
                    




if __name__ == '__main__':
    dataset_path = ""
    dataset = "FB15k-237N"
    kge_model_path = ""
    get_entity_candidate(kge_model_path, dataset_path, dataset)

            

    
