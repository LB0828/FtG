import os


def transform_data(data_path):
    entity_dict = {}
    with open(os.path.join(data_path, "entity2id.txt"), "r") as f:
        lines = f.readlines()
        num_entity = lines[0]
        for line in lines[1:]:
            entity, entity_id = line.strip().split("\t")
            entity_dict[int(entity_id)] = entity
        assert len(entity_dict.keys()) == int(num_entity), "entity num error"
    with open(os.path.join(data_path, "transformer_data", "entities.dict"), "w") as f:
        for entity_id in entity_dict.keys():
            f.write("{}\t{}\n".format(entity_id, entity_dict[entity_id]))
    relation_dict = {}
    with open(os.path.join(data_path, "relation2id.txt"), "r") as f:
        lines = f.readlines()
        num_relation = lines[0]
        for line in lines[1:]:
            relation, relation_id = line.strip().split("\t")
            relation_dict[int(relation_id)] = relation
        assert len(relation_dict.keys()) == int(num_relation), "relation num error"
    with open(os.path.join(data_path, "transformer_data", "relations.dict"), "w") as f:
        for relation_id in relation_dict.keys():
            f.write("{}\t{}\n".format(relation_id, relation_dict[relation_id]))
    # transform the train, valid and test data
    for split in ['train', 'valid', 'test']:
        with open(os.path.join(data_path, "{}2id.txt".format(split)), "r") as f:
            lines = f.readlines()
            num_triple = lines[0]
            with open(os.path.join(data_path, "transformer_data", "{}.txt".format(split)), "w") as f:
                for line in lines[1:]:
                    head, tail, relation = line.strip().split(" ")
                    f.write("{}\t{}\t{}\n".format(entity_dict[int(head)], relation_dict[int(relation)], entity_dict[int(tail)]))


if __name__ == "__main__":
    transform_data("")