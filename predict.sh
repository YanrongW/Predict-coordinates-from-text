PREDICT_DIR=/nfs/volume-93-2/xjl/temp_data
SAVE_FILE_DIR=/nfs/volume-93-1/wangyanrong/poi_generation/data/pre_data

CUDA_VISIBLE_DEVICES=2 python train.py --task_name generation \
      --predict_dir $PREDICT_DIR \
      --resume_dir /nfs/volume-93-1/wangyanrong/poi_generation/generation_model_bart_0530 \
      --resume \
      --do_predict \
      --model_best_or_last model_best.tar \
      --predict_save_file $SAVE_FILE_DIR \
      --output_dir /nfs/volume-93-1/wangyanrong/poi_generation/generation_model_bart \
      --data_dir /nfs/volume-93-1/wangyanrong/poi_generation/data/poi_word_compts_beijing_filter_s2 \
      --eval_dir /nfs/volume-93-1/wangyanrong/poi_generation/data/part-00000_s2 \
      --val_ratio 0.001 \
      --cache_dir /nfs/volume-93-1/wangyanrong/poi_generation/pretrain \
      --decoder_vocab ./decoder_vocab \
      --encoder_vocab ./encoder_vocab.txt \
      --max_seq_length 64 \
      --label_length 2 \
      --train_batch_size 32 \
      --eval_batch_size 16 \
      --evaluation_steps 100 \
      --learning_rate 1e-5 \
      --num_train_epochs 1 \
      --saved_logging_steps 300