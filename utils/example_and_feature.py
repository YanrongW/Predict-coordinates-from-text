import numpy as np
import torch


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, origin_tokens=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.origin_tokens = origin_tokens


class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, position_ids, label, origin_tokens=None,
                 decoder_input_ids=None, decoder_attention_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.position_ids = position_ids
        self.label = label
        self.origin_tokens = origin_tokens
        self.decoder_input_ids = decoder_input_ids
        self.decoder_attention_mask = decoder_attention_mask



def get_position_and_mask_matrix(tokens, max_length):
    """
    根据token的绝对位置，实现掩码矩阵的构建
    :param tokens:  ['x', 'y']
    :param max_seq_length: int
    :return:
    """
    pos = []  # 软位置
    abs = []  # 原始token的绝对位置
    final_token = []
    out_token = []  # 要打印出来的token
    seg = []
    flag_sep = 0  # 是否遇到sep
    pos_idx = -1
    abs_idx = -1
    i = 0
    while i < len(tokens):
        token = tokens[i]
        out_token.append(token)
        final_token.append(token)
        if token == '[SEP]':
            flag_sep = 1
        if flag_sep:
            seg += [1]
        else:
            seg += [0]
        abs_idx += 1
        pos_idx += 1
        abs += [abs_idx]
        pos += [pos_idx]
        i += 1

    token_num = len(final_token)
    visible_matrix = np.zeros((token_num, token_num))
    input_mask = [1] * token_num
    for i in range(token_num):
        visible_abs_idx = abs
        visible_matrix[i, visible_abs_idx] = 1
    if token_num < max_length:
        pad_num = max_length - token_num
        final_token += ['[PAD]'] * pad_num
        seg += [0] * pad_num
        pos += [max_length - 1] * pad_num
        input_mask += [0] * pad_num
        visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
    else:
        final_token = final_token[:max_length]
        seg = seg[:max_length]
        pos = pos[:max_length]
        input_mask = input_mask[:max_length]
        visible_matrix = visible_matrix[:max_length, :max_length]
    if len(out_token) < max_length:
        out_token += ['[PAD]'] * (max_length-len(out_token))
    else:
        out_token = out_token[:max_length]
    return final_token, seg, pos, input_mask, out_token


def get_input_segments(tokens_a, tokens_b, tokenizer, max_seq_length):
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    tokens, segment_ids, position_ids, input_mask, out_token = get_position_and_mask_matrix(tokens, max_seq_length)
    # 将分词后的token映射为数字
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # decoder_tokens = tokenizer.tokenize(example.label)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(position_ids) == max_seq_length
    return input_ids, input_mask, segment_ids, position_ids, out_token


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_example_to_feature(example, max_seq_length, tokenizer, print_info=True):
    """Loads a data file into a list of `InputBatch`s."""
    # 将text_a 与 text_b 里面的名称和地址字段划分为三个阶段
    example.text_a = [example.text_a]
    example.text_b = [example.text_b]

    if (isinstance(example.text_a, str)):
        tokens_a = tokenizer.tokenize(example.text_a)
    elif (isinstance(example.text_a, list)):
        tokens_a = tokenizer.tokenize(example.text_a[0])
    if (isinstance(example.text_b, str)):
        tokens_b = tokenizer.tokenize(example.text_b)
    elif (isinstance(example.text_b, list)):
        tokens_b = tokenizer.tokenize(example.text_b[0])
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    #####
    input_segments = get_input_segments(tokens_a, tokens_b, tokenizer, max_seq_length)
    input_ids, input_mask, segment_ids, position_ids, tokens = input_segments
    # val = torch.nn.functional.normalize(example.label, dim=0)
    bj_center = [116.39747, 39.908823]
    val = list(map(lambda x: x[0]-x[1], zip(example.label, bj_center)))
    feature = InputFeature(input_ids=input_ids,
                           input_mask=input_mask,
                           segment_ids=segment_ids,
                           position_ids=position_ids,
                           label=val,
                           origin_tokens=tokens)
    return feature


if __name__ == '__main__':
    pass
