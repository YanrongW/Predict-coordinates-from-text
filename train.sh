

CUDA_VISIBLE_DEVICES=1 python train.py --task_name generation \
      --data_dir /nfs/volume-93-1/wangyanrong/poi_generation/data/poi_word_compts_beijing_filter_s2 \
      --eval_dir /nfs/volume-93-1/wangyanrong/poi_generation/data/part-00000_s2 \
      --val_ratio 0.001 \
      --output_dir /nfs/volume-93-2/wangyanrong/poi_generation/generation_model_v1_20220613 \
      --cache_dir /nfs/volume-93-1/wangyanrong/poi_generation/pretrain \
      --encoder_vocab ./encoder_vocab.txt \
      --max_seq_length 64 \
      --label_length 2 \
      --do_train \
      --do_eval \
      --train_batch_size 64 \
      --eval_batch_size 32 \
      --evaluation_steps 100 \
      --learning_rate 1e-3 \
      --num_train_epochs 32 \
      --saved_logging_steps 300 \
      --route_dir /nfs/volume-93-2/wangjun/link_citys/link_recall_name_beijing


# 如果加路网，就加上 参数 --add_route \ --route_num 2 \

# 如果要在之前训练的模型的基础上继续训练，则加上下面两个参数
# --resume_dir /nfs/volume-93-1/wangyanrong/poi_generation/generation_model_bart \
# --resume \