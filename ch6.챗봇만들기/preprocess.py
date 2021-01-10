import os
import re
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from konlpy.tag import Okt

FILTERS = "([~.,!?\"':;)(])"
PAD = "<PAD>" # 패딩 토큰
STD = "<SOS>" # 시작 토큰
END = "<END>" # 종료 토큰
UNK = "<UNK>" # 사전에 없는 토큰

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

MAX_SEQUENCES = 25 # 최대 문장 길이


# 데이터 불러오는 함수
def load_data(path):
    data_df = pd.read_csv(path, header=0)
    question, answer = list(data_df['Q']), list(data_df['A'])

    return question, answer

# 데이터를 전처리한 후 단어 리스트로 만들기 (음절리스트)
def data_tokenizer(data):
    words = []
    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split():
            words.append(word)
    # 토그나이징과 정규표현식을 통해 만들어진 값들 넘겨준다
    return [word for word in words if word]


# 한글 텍스트를 토크나이징 하기 위해 형태소로 분석하는 함수 (형태소단위로 토크나이징)
# 환경 설정 파일을 통해 사용할지 안할지 정한다.
def prepor_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ','')))
        result_data.append(morphlized_seq)

    return result_data

# 단어 사전을 만드는 함수
def load_vocabulary(path, vocab_path, tokenize_as_morph=False):
    vocabulary_list = []
    if not os.path.exists(vocab_path):
        # 이미 생성된 사전 파일이 존재하지 않으므로 데이터를 가지고 만들어야 한다.
        # 그래서 데이터가 존재 하면 사전을 만들기 위해서 데이터 파일의 존재 유무를 확인한다.
        if os.path.exists(path):
            data_df = pd.read_csv(path, encoding='utf-8')
            question, answer = list(data_df['Q']), list(data_df['A'])
            data = []
            # 질문과 답변을 extend을 통해서 구조가 없는 배열로 만든다.
            data.extend(question)
            data.extend(answer)

            words = data_tokenizer(data)
            words = list(set(words)) # 공통적인 단어에 대해서는 중복처리함
            words[:0] = MARKER
        
        with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')
    
    with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())

    word2idx, idx2word = make_vocabulary(vocabulary_list)

    return word2idx, idx2word, len(word2idx)


def make_vocabulary(vocabulary_list):
    word2idx = {word: idx for idx, word in enumerate(vocabulary_list)}
    idx2word = {idx: word for idx, word in enumerate(vocabulary_list)}

    return word2idx, idx2word


# 인코더에 대한 전처리 (인코더에 적용될 입력값을 만드는 함수)
def enc_processing(value, dictionary, tokenize_as_morph=False):
    sequences_input_index = []
    sequences_length = []

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_idx = []
        
        for word in sequence.split():
            if dictionary.get(word) is not None: # 단어 사전에 있으면 단어 인덱스
                sequence_idx.extend([dictionary[word]])
            else: # 없으면 UNK 인덱스
                sequence_idx.extend([dictionary[UNK]])

        if len(sequence_idx) > MAX_SEQUENCES:
            sequence_idx = sequence_idx[:MAX_SEQUENCES] # 최대 문장 길이보다 길면 잘라줌

        sequences_length.append(len(sequence_idx)) # 하나의 문장에 길이를 넣어주고 있다.
        sequence_idx += (MAX_SEQUENCES - len(sequence_idx)) * [dictionary[PAD]] # 남은 길이만큼 패딩으로 채워줌

        sequences_input_index.append(sequence_idx)

    # 인덱스화된 일반 배열을 넘파이 배열로 변경한다.
    # 이유는 텐서플로우 dataset에 넣어 주기 위한 사전 작업이다.
    # 넘파이 배열에 인덱스화된 배열과 패딩 처리 전의 각 문장의 실제 길이를 넘겨준다.
    return np.asarray(sequences_input_index), sequences_length


# 디코더 입력값을 만드는 함수
def dec_output_processing(value, dictionary, tokenize_as_morph=False):
    sequences_output_index = []
    sequences_length = []

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, '', sequence)
        sequence_idx = []
        sequence_idx = [dictionary[STD]] + [dictionary[word] for word in sequence.split()] # 시작 토큰 넣어주기

        if len(sequence_idx) > MAX_SEQUENCES:
            sequence_idx = sequence_idx[:MAX_SEQUENCES]
        
        sequences_length.append(len(sequence_idx))
        sequence_idx += (MAX_SEQUENCES - len(sequence_idx)) * [dictionary[PAD]]

        sequences_output_index.append(sequence_idx)

    return np.asarray(sequences_output_index), sequences_length

# 디코더 타깃값을 만드는 함수

def dec_target_processing(value, dictionary, tokenize_as_morph=False):
    # 인덱스 값들을 가지고 있는
    # 배열이다.(누적된다)
    sequences_target_index = []
    # 형태소 토크나이징 사용 유무
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)
    # 한줄씩 불어온다.
    for sequence in value:
        # FILTERS = "([~.,!?\"':;)(])"
        # 정규화를 사용하여 필터에 들어 있는
        # 값들을 "" 으로 치환 한다.
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        # 문장에서 스페이스 단위별로 단어를 가져와서
        # 딕셔너리의 값인 인덱스를 넣어 준다.
        # 디코딩 출력의 마지막에 END를 넣어 준다.
        sequence_index = [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]
        # 문장 제한 길이보다 길어질 경우 뒤에 토큰을 자르고 있다.
        # 그리고 END 토큰을 넣어 준다
        if len(sequence_index) >= MAX_SEQUENCES:
            sequence_index = sequence_index[:MAX_SEQUENCES - 1] + [dictionary[END]]
        else:
            sequence_index += [dictionary[END]]
        # max_sequence_length보다 문장 길이가 작다면 빈 부분에 PAD(0)를 넣어준다.
        sequence_index += (MAX_SEQUENCES - len(sequence_index)) * [dictionary[PAD]]
        # 인덱스화 되어 있는 값을 sequences_target_index에 넣어 준다.
        sequences_target_index.append(sequence_index)
    return np.asarray(sequences_target_index)

