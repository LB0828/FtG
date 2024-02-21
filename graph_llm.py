import torch.nn as nn
import torch


class GraphLLM(nn.Module):
    def __init__(self, llama_model, kge_model, kge_model_path):
        super(GraphLLM, self).__init__()
        self.llama_model = llama_model
        self.ent_embs, self.rel_embs = self.load_pretraining_kge(kge_model, kge_model_path)
        
        self.projector = nn.Sequential(
            nn.Linear(self.ent_embs.shape[1], 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096),
        )
    
    def forward(self, input_ids, attention_mask, graph_tokens, labels):
        batch_size, prefix_len = graph_tokens.shape
        ent_token_embs = self.ent_embs[graph_tokens[:][::2]]
        rel_token_embs = self.rel_embs[graph_tokens[:][1::2]]
        graph_tokens_list = []
        for i in range(rel_token_embs.shape[0]):
            graph_tokens_list.append(ent_token_embs[:][i])
            graph_tokens_list.append(rel_token_embs[:][i])
        graph_tokens_list.append(ent_token_embs[:][-1])
        graph_tokens_emb = torch.stack(graph_tokens_list, dim=1)
        graph_tokens_emb = self.projector(graph_tokens_emb)
        token_embs = self.llama_model.model.model.embed_tokens(input_ids)
        input_embs = torch.cat((graph_tokens_emb, token_embs), dim=1)
        prefix_mask = torch.ones((batch_size, prefix_len))
        prefix_labels = torch.full((batch_size, prefix_len), -100)
        new_attention_mask = torch.cat((prefix_mask.cuda(), attention_mask), dim=-1)
        new_labels = torch.cat((prefix_labels.cuda(), labels), dim=-1)

        return self.llama_model(input_embs, new_attention_mask, new_labels)

    def load_pretraining_kge(self, kge_model, kge_model_path):
        kge_model = torch.load(kge_model_path)
        ent_embs = kge_model.entity_embedding
        rel_embs = kge_model.relation_embedding
        ent_embs.requires_grad = False
        rel_embs.requires_grad = False
        ent_dim = ent_embs.shape[1]
        rel_dim = rel_embs.shape[1]
        if ent_dim != rel_dim:
            rel_embs = torch.cat((rel_embs, rel_embs), dim=-1)
        return ent_embs, rel_embs
