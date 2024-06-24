import torch
import os
from collections import defaultdict
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pickle

def read_triples(dataset_path, dataset, filename):
    file_path = os.path.join(dataset_path, dataset, filename)
    with open(file_path) as file:
        lines = file.read().strip().split('\n')
    triples = []
    for line in lines[1:]:
        split = list(map(int, line.split(' ')))
        triples.append(split)
    return triples

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
    # rel_name_list = [name.split(' , ')[-1].strip() for name in rel_name_list]
    return ent_name_list, rel_name_list

def prune_subgraph(triples, head_entity_1hop_neighbor, tail_entity_1hop_neighbor, ent_emb, rel_emb):
    query_tail_neighbors = {}
    query_head_neighbors = {}
    cos = nn.CosineSimilarity(dim=1)
    for idx, triple in tqdm(enumerate(triples), total=len(triples)):
        head, tail, rel = triple
        head_related_triples = head_entity_1hop_neighbor.get(head, [])
        if (rel, tail) in head_related_triples:
            head_related_triples.remove((rel, tail))
        if len(head_related_triples) <= 3:
            head_neighbors = head_related_triples + [(-500, -500)] * (3 - len(head_related_triples))
            head_neighbors = np.array(head_neighbors).flatten().tolist()
        else:
            head_neighbors = torch.cat([torch.tensor([head] * len(head_related_triples)).unsqueeze(-1),
                                        torch.tensor(head_related_triples)], dim=-1)
            test_head_rel_emb = torch.cat([ent_emb[head], rel_emb[rel]], dim=-1).unsqueeze(0)
            head_neighbors_emb = torch.cat([ent_emb[head_neighbors[:, 0]], rel_emb[head_neighbors[:, 1]]], dim=-1)
            similarity = cos(test_head_rel_emb, head_neighbors_emb)
            _, top3_idx = torch.topk(similarity, 3)
            head_neighbors = torch.tensor(head_related_triples)[top3_idx].flatten().tolist()
        query_tail_neighbors[idx] = [head] + head_neighbors
        # query head neighbors
        tail_related_triples = tail_entity_1hop_neighbor.get(tail, [])
        if (head, rel) in tail_related_triples:
            tail_related_triples.remove((head, rel))
        if len(tail_related_triples) < 3 and len(tail_related_triples) > 0:
            tail_neighbors = tail_related_triples + [(-500, -500)] * (3 - len(tail_related_triples))
            tail_neighbors = np.array(tail_neighbors).flatten().tolist()
            tail_neighbors.insert(tail_neighbors.index(-500), tail)
        elif len(tail_related_triples) == 0:
            tail_neighbors = [-500] * 6 + [tail]
        else:
            tail_neighbors = torch.cat([torch.tensor(tail_related_triples),
                                        torch.tensor([tail] * len(tail_related_triples)).unsqueeze(-1)], dim=-1)
            test_rel_tail_emb = torch.cat([rel_emb[rel], ent_emb[tail]], dim=-1).unsqueeze(0)
            tail_neighbors_emb = torch.cat([rel_emb[tail_neighbors[:, 1]], ent_emb[tail_neighbors[:, 2]]], dim=-1)
            similarity = cos(test_rel_tail_emb, tail_neighbors_emb)
            _, top3_idx = torch.topk(similarity, 3)
            tail_neighbors = torch.tensor(tail_related_triples)[top3_idx].flatten().tolist() + [tail]
        query_head_neighbors[idx] = tail_neighbors
        

    test_triples_pruned_neighbors = {
        'query_tail_neighbors': query_tail_neighbors,
        'query_head_neighbors': query_head_neighbors
    }
    return test_triples_pruned_neighbors

def get_graph_emb(triples, head_entity_1hop_neighbor, tail_entity_1hop_neighbor, ent_emb, rel_emb, graph_emb, ent_name_list, rel_name_list):
    query_head_neighbors = {}
    query_tail_neighbors = {}
    context_list = ["" for _ in range(len(triples) * 2)]
    cos = nn.CosineSimilarity(dim=1)
    for idx, triple in tqdm(enumerate(triples), total=len(triples)):
        head, tail, rel = triple
        head_related_triples = head_entity_1hop_neighbor.get(head, [])
        if (rel, tail) in head_related_triples:
            head_related_triples.remove((rel, tail))
        if len(head_related_triples) <= 3 and len(head_related_triples) > 0:
            head_neighbors = np.array(head_related_triples)[:, 1].tolist()
            context = head_related_triples
        elif len(head_related_triples) == 0:
            head_neighbors = []
            context = []
        else:
            head_neighbors = torch.cat([torch.tensor([head] * len(head_related_triples)).unsqueeze(-1),
                                        torch.tensor(head_related_triples)], dim=-1)
            test_head_rel_emb = torch.cat([ent_emb[head], rel_emb[rel]], dim=-1).unsqueeze(0)
            head_neighbors_emb = torch.cat([ent_emb[head_neighbors[:, 0]], rel_emb[head_neighbors[:, 1]]], dim=-1)
            similarity = cos(test_head_rel_emb, head_neighbors_emb)
            _, top3_idx = torch.topk(similarity, 3)
            head_neighbors = torch.tensor(head_related_triples)[top3_idx][:, 1].tolist()
            context = torch.tensor(head_related_triples)[top3_idx].tolist()
        head_neighbors = [head] + head_neighbors
        head_neighbors_emb = ent_emb[head_neighbors].mean(dim=0)
        graph_emb[idx + len(triples)] = head_neighbors_emb
        for rel, tail in context:
            context_list[idx + len(triples)] += ent_name_list[head] + ' ' + rel_name_list[rel] + ' ' + ent_name_list[tail] + ' '

        # get tail related neighbors
        tail_related_triples = tail_entity_1hop_neighbor.get(tail, [])
        if (head, rel) in tail_related_triples:
            tail_related_triples.remove((head, rel))
        if len(tail_related_triples) <= 3 and len(tail_related_triples) > 0:
            tail_neighbors = np.array(tail_related_triples)[:, 0].tolist()
            context = tail_related_triples
        elif len(tail_related_triples) == 0:
            tail_neighbors = []
            context = []
        else:
            tail_neighbors = torch.cat([torch.tensor(tail_related_triples),
                                        torch.tensor([tail] * len(tail_related_triples)).unsqueeze(-1)], dim=-1)
            test_rel_tail_emb = torch.cat([rel_emb[rel], ent_emb[tail]], dim=-1).unsqueeze(0)
            tail_neighbors_emb = torch.cat([rel_emb[tail_neighbors[:, 1]], ent_emb[tail_neighbors[:, 2]]], dim=-1)
            similarity = cos(test_rel_tail_emb, tail_neighbors_emb)
            _, top3_idx = torch.topk(similarity, 3)
            tail_neighbors = torch.tensor(tail_related_triples)[top3_idx][:, 0].flatten().tolist()
            context = torch.tensor(tail_related_triples)[top3_idx].tolist()

        tail_neighbors = tail_neighbors + [tail]
        tail_neighbors_emb = ent_emb[tail_neighbors].mean(dim=0)
        graph_emb[idx] = tail_neighbors_emb
        for head, rel in context:
            context_list[idx] += ent_name_list[head] + ' ' + rel_name_list[rel] + ' ' + ent_name_list[tail] + ' '
        
    return graph_emb, context_list

        


def get_entity_1hop_neighbor(data_path, dataset, ent_emb, rel_emb):
    head_entity_1hop_neighbor = defaultdict(list)
    tail_entity_1hop_neighbor = defaultdict(list)
    train_triples = read_triples(data_path, dataset, 'train2id.txt')
    test_triples = read_triples(data_path, dataset, 'test2id.txt')
    ent_name_list, rel_name_list = read_name(data_path, dataset)

    for triple in train_triples:
        head, tail, rel = triple
        head_entity_1hop_neighbor[head].append((rel, tail))
        tail_entity_1hop_neighbor[tail].append((head, rel))
    print('get train_triples 1-hop pruned neighbors')
    train_graph_emb = torch.zeros((len(train_triples) * 2, ent_emb.shape[1]))
    train_graph_emb, train_context_list = get_graph_emb(train_triples, head_entity_1hop_neighbor, tail_entity_1hop_neighbor, ent_emb, rel_emb, train_graph_emb, ent_name_list, rel_name_list)
    torch.save(train_graph_emb, os.path.join(data_path, dataset, 'train_graph_emb.pt'))
    with open(os.path.join(data_path, dataset, 'train_context_list.pkl'), 'wb') as file:
        pickle.dump(train_context_list, file)
    test_graph_emb = torch.zeros((len(test_triples) * 2, ent_emb.shape[1]))
    test_graph_emb, test_context_list = get_graph_emb(test_triples, head_entity_1hop_neighbor, tail_entity_1hop_neighbor, ent_emb, rel_emb, test_graph_emb, ent_name_list, rel_name_list)
    torch.save(test_graph_emb, os.path.join(data_path, dataset, 'test_graph_emb.pt'))
    with open(os.path.join(data_path, dataset, 'test_context_list.pkl'), 'wb') as file:
        pickle.dump(test_context_list, file)



    # train_triples_pruned_neighbors = prune_subgraph(train_triples, head_entity_1hop_neighbor, tail_entity_1hop_neighbor, ent_emb, rel_emb)
    # print('get test_triples 1-hop pruned neighbors')
    # test_triples_pruned_neighbors = prune_subgraph(test_triples, head_entity_1hop_neighbor, tail_entity_1hop_neighbor, ent_emb, rel_emb)

    # with open(os.path.join(data_path, dataset, 'train_triples_pruned_neighbors.pkl'), 'wb') as file:
    #     pickle.dump(train_triples_pruned_neighbors, file)
    
    # with open(os.path.join(data_path, dataset, 'test_triples_pruned_neighbors.pkl'), 'wb') as file:
    #     pickle.dump(test_triples_pruned_neighbors, file)
    




if __name__ == '__main__':
    emb_path = './FtG/KnowledgeGraphEmbedding/models/RotatE_codex-m_0'
    ent_emb = torch.from_numpy(np.load(os.path.join(emb_path, 'entity_embedding.npy')))
    rel_emb = torch.from_numpy(np.load(os.path.join(emb_path, 'relation_embedding.npy')))
    data_path = './FtG/data/processed'
    dataset = 'codex-m'
    get_entity_1hop_neighbor(data_path, dataset, ent_emb, rel_emb)