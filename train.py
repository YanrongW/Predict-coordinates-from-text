# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import math
import os
import random, time
import numpy as np
import s2sphere
import torch
from transformers import MT5ForConditionalGeneration, MT5Config, MT5Model
# from transformers import XLNetModel, XLNetConfig
from transformers import MBartModel, MBartConfig
from transformers import BertModel, BertConfig
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from model import RBertForSequenceClassification
import torch.nn as nn
from utils.dataset import BaseDataset, RelationDataset, RelationDataLoader
from sklearn import metrics
# torch.backends.cudnn.deterministic = True
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
logger = logging.getLogger(__name__)


def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def _calc_distance(slng, slat, dlng, dlat):
    PI = 3.14159265358979
    RAD = PI / 180.0
    EARTH_RADIUS = 6378137
    radLat1 = slat * RAD
    radLat2 = dlat * RAD
    a = radLat1 - radLat2
    b = (slng - dlng) * RAD
    s = 2 * math.asin(
        math.sqrt(pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * pow(math.sin(b / 2), 2)))
    dis = s * EARTH_RADIUS
    return dis


def classification_train(args):
    route_data = None
    # ------读取全部北京POI数据--------
    print('---  load all beijing data -----')
    train_data = read_data(args.data_dir) # 训练数据
    val_data = read_data(args.eval_dir) # 验证数据
    print('==========')
    print('train_data: ', len(train_data))
    print('val_data: ', len(val_data))
    # 构造训练数据 poi_id, name, address, lng, lat, s2_code
    train_dataset = RelationDataset(args=args, all_data=train_data)
    train_data = RelationDataLoader(args, train_dataset)
    train_dataloader = train_data.get_dataloader()
    print("train_dataloader ready!")
    # 构造验证数据
    val_dataset = RelationDataset(args=args, all_data=val_data)
    val_data = RelationDataLoader(args, val_dataset)
    val_dataloader = val_data.get_dataloader(is_test=True)
    print("val_dataloader ready!")
    num_train_optimization_steps = int(
        len(train_data.examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    # Prepare model
    # config = XLNetConfig.from_pretrained('xlnet-base-cased', cache_dir=args.cache_dir, num_labels=1)
    # model = RXLNetForSequenceClassification.from_pretrained('xlnet-base-cased', cache_dir=args.cache_dir,
    #                                                        config=config)
    # model = RMBartForSequenceClassification.from_pretrained('facebook/mbart-large-cc25', cache_dir=args.cache_dir,
    #                                                        config=config)
    # ------ Bert ------
    # config = BertConfig.from_pretrained('bert-base-chinese', cache_dir=args.cache_dir, num_labels=1)
    # config.vocab_size = train_dataset.tokenizer.vocab_size
    # args.bert_config = config
    # model = RBertForSequenceClassification.from_pretrained('bert-base-chinese', cache_dir=args.cache_dir,
    #                                                        config=config)
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=1)
    config.vocab_size = train_dataset.tokenizer.vocab_size
    args.bert_config = config
    model = RBertForSequenceClassification.from_pretrained('bert-base-chinese', cache_dir=args.cache_dir,
                                                           config=config)
    # 学习率调整策略
    if args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)
        warm_up_steps = int(args.warmup_proportion * num_train_optimization_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps,
                                                    num_training_steps=num_train_optimization_steps)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=0, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    best_dis = None
    weights = None
    check_point_path = os.path.join(args.resume_dir, "model_best.tar")
    start_epoch, start_step, global_step = 0, 0, 1
    if os.path.exists(check_point_path) and args.resume:
        check_point = torch.load(check_point_path)
        model.load_state_dict(check_point['model_state_dict'])
        model.to(args.device)
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        start_step = check_point['step']
        start_epoch = check_point['epoch']
        global_step = check_point['global_step']
        logger.info("Load model from {} in step {}".format(check_point_path, check_point['step']))
        logger.info("Load model from {} in epoch {}".format(check_point_path, check_point['epoch']))
        eval_loss, all_predictions, all_gold_answers, ave_dis, all_dis, pred_geo, true_geo = classification_eval(model, val_dataloader,
                                                                                                      device=args.device,
                                                                                                      args=args,
                                                                                                      tokenizer=train_dataset.tokenizer)
        best_dis = ave_dis
    model.to(args.device)

    model.train()
    now = int(time.time())
    timeArray = time.localtime(now)
    nowtime = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)
    results = {}
    not_change_best_num = 0
    total_loss_lst = []
    for epoch in trange(start_epoch, int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        if len(total_loss_lst) == 0:
            average_loss = 0
        nb_tr_steps, nb_tr_examples = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            b1 = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, position_ids, label_ids = b1
            '''输入BERT前一定要注意格式'''
            input_ids = input_ids.view(-1, input_ids.size(-1))
            input_mask = input_mask.view(-1, input_mask.size(-1))
            segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            position_ids = position_ids.view(-1, position_ids.size(-1))
            label_ids = label_ids.view(-1, label_ids.size(-1))
            model_outputs = model(input_ids=input_ids, position_ids=position_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                  labels=label_ids)
            
            # print(model_outputs)
            total_loss, logits = model_outputs[:2]
            nb_tr_examples += input_ids.size(0)
            if args.gradient_accumulation_steps > 1:
                total_loss = total_loss / args.gradient_accumulation_steps
            '''梯度清零'''
            optimizer.zero_grad()
            total_loss.backward()
            tr_loss += total_loss.item()
            # ave_loss = tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1)
            # average_loss = step / (step + 1) * average_loss + total_loss.item() / (step + 1)
            if len(total_loss_lst) > 10000:
                average_loss = sum(total_loss_lst[-10000:]) / (10000 + 1) + total_loss.item() / (10000 + 1)
            else:
                average_loss = step / (step + 1) * average_loss + total_loss.item() / (step + 1)
            total_loss_lst.append(total_loss.item())
            nb_tr_steps += 1
            # average_loss = (global_step - 1) / global_step * average_loss + total_loss.item() / global_step
            logger.info("Step[{}/{}] total_loss={}, ave_loss={}, best_dis={}, not_change_best_num={}, labels={}, logits={}".
                        format(step, len(train_dataloader), round(total_loss.item(), 3),
                               round(average_loss, 3), best_dis if best_dis is not None else -1, not_change_best_num,
                               label_ids, logits))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                '''更新参数'''
                optimizer.step()
                scheduler.step()
                global_step += 1
            if (global_step + 1) % args.saved_logging_steps == 0:
                logger.info("Global_step {}: Save the states!".format(str(global_step)))
                torch.save({
                    "global_step": global_step,
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, os.path.join(args.output_dir, "model_last.tar"))
                torch.save(model.state_dict(), os.path.join(args.output_dir, "model_last.pt"))
                logger.info('Save model_last !')
                if 'loss' in results:
                    results['loss'].append(average_loss)
                else:
                    results['loss'] = []

            if (global_step + 1) % args.evaluation_steps == 0 and args.do_eval:
                logger.info("***** Running evaluation *****")
                eval_loss, all_predictions, all_gold_answers, ave_dis, all_dis, pred_geo, true_geo = classification_eval(model, val_dataloader,
                                                                                                              device=args.device, args=args,
                                                                                                              tokenizer=train_dataset.tokenizer)
                if best_dis is None or best_dis > ave_dis:
                    best_dis = ave_dis
                    torch.save({
                        "global_step": global_step,
                        "epoch": epoch,
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()
                    }, os.path.join(args.output_dir, "model_best.tar"))
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_last.pt"))
                    print('Save model_best ! , distance is ', best_dis)
                else:
                    not_change_best_num += 1
                pass


# 预测经纬度&真实经纬度 计算距离
def calculate_ave_distance(logits, labels, device):
    bj_center = torch.tensor([116.39747, 39.908823], dtype=torch.float)
    bj_center = bj_center.to(device)
    logits = list(map(lambda x:x+bj_center, logits))
    labels = list(map(lambda x:x+bj_center, labels))
    ans_distance = 0.0
    pred_geo = []
    true_geo = []
    mi_dis = 0
    mx_dis = 0
    same_num = 0
    all_dis = []
    all_geo = []
    num_low_dic = {100: 0,300: 0, 500: 0, 1000: 0, 2000: 0, 3000: 0}
    for i in tqdm(range(len(logits)), desc='calculate_ave_distance'):
        now_geo = logits[i]
        origin_geo = labels[i]
        pred_geo.append(now_geo)
        true_geo.append(origin_geo)
        now_dis = _calc_distance(slng=float(origin_geo[0]), slat=float(origin_geo[1]),
                                 dlng=float(now_geo[0]), dlat=float(now_geo[1]))
        if now_dis < 100:
            num_low_dic[100] += 1
        if now_dis < 300:
            num_low_dic[300] += 1
        if now_dis < 500:
            num_low_dic[500] += 1
        if now_dis < 1000:
            num_low_dic[1000] += 1
        if now_dis < 2000:
            num_low_dic[2000] += 1
        if now_dis < 3000:
            num_low_dic[3000] += 1
        all_dis.append(now_dis)
        all_geo.append(str(now_geo[0])+','+str(now_geo[1]))
        if i == 0:
            mi_dis = now_dis
            mx_dis = now_dis
        else:
            mi_dis = min(mi_dis, now_dis)
            mx_dis = max(mx_dis, now_dis)
        ans_distance += now_dis

    ave_distance = round(ans_distance/len(logits), 10)
    print('====== ans_dis ======')
    print('ave_dis ', ave_distance)
    print('mx_dis: {}, mi_dis: {}'.format(mx_dis, mi_dis))
    print(pred_geo[:10])
    print(true_geo[:10])
    for key in num_low_dic:
        print(key, " ", num_low_dic[key])
    print('all_num: {}, acc: {}'.format(len(pred_geo), round((same_num / len(pred_geo)) * 100, 4)))
    print('all_geo: ', all_geo[:2])
    return ave_distance, all_dis, pred_geo, true_geo


def classification_eval(model, val_dataloader, device, args, tokenizer):
    all_predictions = []
    all_gold_answers = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for step, batch in enumerate(tqdm(val_dataloader, desc="Iteration")):
        model.eval()
        b1 = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = b1
        '''输入BERT前一定要注意格式'''
        input_ids = input_ids.view(-1, input_ids.size(-1))
        input_mask = input_mask.view(-1, input_mask.size(-1))
        segment_ids = segment_ids.view(-1, segment_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))
        label_ids = label_ids.view(-1, label_ids.size(-1)) # batch_size * 2
        with torch.no_grad():
            model_outs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                               labels=label_ids)
            # model_outs = model(input_ids=input_ids, attention_mask=input_mask,
            #                    labels=label_ids)
            tmp_eval_loss, logits = model_outs[:2]

        all_predictions.extend(logits)
        all_gold_answers.extend(label_ids)
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1
    print('relative_pred_vals: ', all_predictions[:10])
    print('relative_ans: ', all_gold_answers[:10])
    print('length of relative_pred_vals: ', len(all_predictions))
    print('length of relative_ori_vals: ', len(all_gold_answers))
    ave_dis, all_dis, pred_geo, true_geo = calculate_ave_distance(all_predictions, all_gold_answers, device)
    model.train()
    eval_loss = eval_loss / nb_eval_steps
    print('eval_loss：', eval_loss)
    return eval_loss, all_predictions, all_gold_answers, ave_dis, all_dis, pred_geo, true_geo


def classification_predict(args):
    print('------ load all beijing data ------')
    test_data = read_data(args.predict_dir)
    print('test_data: ', len(test_data))

    predict_dataset = RelationDataset(args=args, all_data=test_data)
    predict_data = RelationDataLoader(args, predict_dataset)
    predict_dataloader = predict_data.get_dataloader(is_test=True)
    print("predict_dataloader ready!")

    config = BertConfig.from_pretrained('bert-base-chinese', cache_dir=args.cache_dir, num_labels=1)
    config.vocab_size = predict_dataset.tokenizer.vocab_size
    args.bert_config = config
    model = RBertForSequenceClassification.from_pretrained('bert-base-chinese', cache_dir=args.cache_dir,
                                                            config=config)
    if args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=0, amsgrad=False)

    check_point_path = os.path.join(args.resume_dir, args.model_best_or_last)
    if os.path.exists(check_point_path) and args.resume:
        check_point = torch.load(check_point_path)
        model.load_state_dict(check_point['model_state_dict'])
        model.to(args.device)
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        logger.info("Load model from {} in step {} in epoch {}".format(check_point_path, check_point['global_step'],
                                                                       check_point['epoch']))
    model.to(args.device)

    model.eval()
    eval_loss, all_predictions, gold_answers, ave_dis, all_dis, pred_geo, true_geo = classification_eval(model, predict_dataloader,
                                                                                                        device=args.device, args=args,
                                                                                                        tokenizer=predict_dataset.tokenizer)
    save_data = []
    for i in tqdm(range(len(test_data)), desc='save_pre_data'):
        one_data = test_data[i]
        one_data = json.loads(one_data)
        geo = torch.tensor([float(item) for item in one_data['geo'].split(',')]).to(args.device)
        origin_geo = [float(item) for item in one_data['geo'].split(',')]
        if 'acc_geo' in one_data.keys():
            acc_geo = [float(item) for item in one_data['acc_geo'].split(',')]
            one_data['pre_acc_dis'] = _calc_distance(slng=float(origin_geo[0]), slat=float(origin_geo[1]),
                                                     dlng=float(acc_geo[0]), dlat=float(acc_geo[1]))

        assert np.round(geo.detach().cpu().numpy(), 3).all() == np.round(true_geo[i].detach().cpu().numpy(), 3).all()
        one_data['pre_dis'] = all_dis[i]
        one_data['pre_geo'] = ','.join([str(item) for item in pred_geo[i].detach().cpu().numpy()])
        save_data.append(one_data)
    save_data = sorted(save_data, key=lambda x: x['pre_dis'])
    with open(args.predict_save_file, 'w', encoding='utf-8') as f:
        for one_data in save_data:
            f.write(json.dumps(one_data, ensure_ascii=False) + '\n')
    print('------ Test Finished ------')


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir",
                        default='./data',
                        type=str,
                        # required=True,
                        help="训练数据")
    parser.add_argument("--eval_dir",
                        default='./data',
                        type=str,
                        # required=True,
                        help="训练数据")
    parser.add_argument('--predict_dir',
                        default='',
                        type=str,
                        help='预测文件路径')
    parser.add_argument('--id_file',
                        default='',
                        type=str,
                        help='预测poiiid文件路径')
    parser.add_argument("--val_ratio",
                        type=float,
                        default=0,
                        help="验证集占训练数据的比例")

    parser.add_argument("--bert_model", default="hfl/chinese-bert-wwm-ext", type=str,
                        help='预训练模型地址')
    parser.add_argument("--num_hidden_layers",
                        default=12,
                        type=int,
                        help="The hidden layers in Bert Encoder.")
    parser.add_argument("--task_name",
                        default='genration',
                        type=str,
                        # required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='./output',
                        type=str,
                        # required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--resume_dir",
                        default="",
                        type=str,
                        help="The resume directory where the model predictions and checkpoints saved in.")

    # Other parameters
    basedir = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument("--cache_dir",
                        default=os.path.join(basedir, "pretrain"),
                        type=str,
                        help="Where do you want to store the pre-trained own_models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_ans_length",
                        default=20,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--resume",
                        action='store_true',
                        help="Whether to load the history model.")

    parser.add_argument("--do_train",
                        # default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-2,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--optimizer",
                        default="adamw",
                        type=str,
                        help="optimizer method with different scheduler method.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--saved_logging_steps",
                        default=300,
                        type=int,
                        help=".")
    parser.add_argument("--evaluation_steps",
                        default=1,
                        type=int,
                        help=".")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--route_num', type=int, default=2, help='加入的路网信息的条数')
    parser.add_argument('--add_route', action='store_true', help='是否在数据中加入路网信息')
    parser.add_argument('--route_dir', type=str, default='', help='处理好的路网信息所在路径')
    parser.add_argument('--decoder_vocab', type=str, default='./decoder_vocab', help='解码时用的词典')
    parser.add_argument('--encoder_vocab', type=str, default='./encoder_vocab.txt', help='编码时用的词典')
    parser.add_argument('--weight_loss', action='store_true', help='编码时用的词典')
    parser.add_argument('--label_length', type=int, default=16, help='训练计算损失的时候，标签的长度')
    # predict
    parser.add_argument('--model_best_or_last', type=str, default="model_best.tar", help='预测的时候加载bset还是last')
    parser.add_argument('--predict_save_file', type=str, default='', help='预测文件的保存路径')
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    # logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    #     args.device, n_gpu, bool(args.local_rank != -1), args.fp16))
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    args.seed = random.randint(1, 200)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        classification_train(args)
    elif args.do_predict:
        print('-------predicting-----')
        classification_predict(args)
    else:
        print('At least one of `do_train` or `do_predict` must be True.')


if __name__ == "__main__":
    main()
