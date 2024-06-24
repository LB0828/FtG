# <center> Filter-then-Generate: : Large Language Models with Structure-Text Adapter for Knowledge Graph Completion</center>


<center>Anonymous ACL submission</center>


#### Step 1: Environment Preparation 

```shell
# create a new environment
conda create -n ftg python=3.10
conda activate ftg

# install pytorch. Modify the command to align with your own CUDA version.
pip3 install torch  --index-url https://download.pytorch.org/whl/cu118

# install related libraries
pip install -r requirements.txt

# install pyg
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```
#### Step 2:  Data Preparation
To prepare the data, you can run the `./construct_subgraph/graph_prune.py` and then run `./construct_instruction.py` to construct train and valid instruction datasets.
```
.
├── README.md
├── configs
│   ├── deepspeed_config_stage3.json
│   ├── lora_config_llama.json
├── construct_subgraph
│   ├── graph_prune.py
├── kge
│   ├── codes
│   │   ├── dataloader.py
│   │   ├── model.py
│   │   ├── transform_data.py
│   │   ├── run.py
│   ├── run.sh
├── script
│   ├── train.sh
│   ├── inference.sh
├── utils.py
├── construct_instruction.py
├── lora_ftg.py
├── prompter.py
├── ftg_trainer.py
```
#### Step 3: Training
To execute the training process, you can run either `./scripts/train.sh`. The usage instructions are as follows:
```shell
torchrun --nproc_per_node 4 --master_port 29500 lora_ftg.py \
    --ddp_timeout 36000 \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --use_lora \
    --deepspeed configs/deepspeed_config_stage3.json \
    --lora_config configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 2 \
    --learning_rate 3e-4 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --torch_dtype "bfloat16" \
    --bf16 \
    --seed 1234 \
    --gradient_checkpointing \
    --output_dir ${output_dir} \
```

#### Step 4: Evaluation
You can evaluate FtG with the command:

```shell
ckpt_path="/path/to/projector" # local path or huggingface repo
dataset="FB15k-237" #test dataset #test task
emb="/path/kge_emb"
projector_path='/path/projector.bin'
lora_path='/path/adapter.bin'
output_path="/path/to/output"

python infer.py \
--ckpt_path $ckpt_path \
--lora_path $lora_path \
--infer_file $infer_file \
--llama \
--use_lora \
```

#### Acknowledgement
For conventional kge training, we follow this repo:
```
@inproceedings{
 sun2018rotate,
 title={RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space},
 author={Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang},
 booktitle={International Conference on Learning Representations},
 year={2019},
 url={https://openreview.net/forum?id=HkgEQnRqYQ},
}
```