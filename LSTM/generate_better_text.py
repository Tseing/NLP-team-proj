# coding: utf-8
import sys
sys.path.append('..')
from common.np import *
from rnnlm_gen import BetterRnnlmGen
import preprocess


corpus, word_to_id, id_to_word = preprocess.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)
wordvec_size = 512
hidden_size = 512
dropout = 0.5


model = BetterRnnlmGen(vocab_size, wordvec_size, hidden_size, dropout)
model.load_params('./BetterRnnlm.pkl')

# 设定start字符和skip字符
start_word = '国'
start_id = word_to_id[start_word]

# 文本生成
word_ids = model.generate(start_id)
txt = ''.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')

print(txt)


# model.reset_state()
#
# start_words = 'the meaning of life is'
# start_ids = [word_to_id[w] for w in start_words.split(' ')]
#
# for x in start_ids[:-1]:
#     x = np.array(x).reshape(1, 1)
#     model.predict(x)
#
# word_ids = model.generate(start_ids[-1], skip_ids)
# word_ids = start_ids[:-1] + word_ids
# txt = ' '.join([id_to_word[i] for i in word_ids])
# txt = txt.replace(' <eos>', '.\n')
# print('-' * 50)
# print(txt)
