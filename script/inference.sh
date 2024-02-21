export CUDA_VISIBLE_DEVICES='0'

ckpt_path=
lora_path=
infer_file=
python infer_beam.py \
--ckpt_path $ckpt_path \
--lora_path $lora_path \
--infer_file $infer_file \
--llama \
--use_lora \