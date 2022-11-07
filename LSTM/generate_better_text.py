# coding: utf-8
import sys
sys.path.append('..')
from common.np import *
from rnnlm_gen import BetterRnnlmGen
import preprocess
import thulac


corpus, word_to_id, id_to_word = preprocess.load_data('train', is_multi_file=True, file_idx=0)
vocab_size = len(word_to_id)
corpus_size = len(corpus)
wordvec_size = 512
hidden_size = 512
dropout = 0.5

# print(vocab_size)

model = BetterRnnlmGen(vocab_size, wordvec_size, hidden_size, dropout)
model.load_params('./BetterRnnlm0.pkl')

# # 设定start字符和skip字符
# start_word = '生活'
# start_id = word_to_id[start_word]
#
# # 文本生成
# word_ids = model.generate(start_id)
# txt = ''.join([id_to_word[i] for i in word_ids])
# txt = txt.replace(' <eos>', '.\n')
#
# print(txt)


# model.reset_state()
#

word_sep_model = thulac.thulac(seg_only=True, user_dict="../user_dict.txt")
cnt = 0

# with open("../upload_data.csv", "w", encoding="utf-8") as f:
#     for line in preprocess.get_lines("../dataset/Test_A.csv"):
#         news = line.split(",")[1].strip()
#         start_sentence = news.split(sep=r"[SEP]")[0]
#
#         start_words = np.array(word_sep_model.cut(start_sentence))[::, 0]
#         start_ids = [word_to_id[w] for w in start_words]
#
#         for x in start_ids[:-1]:
#             x = np.array(x).reshape(1, 1)
#             model.predict(x)
#
#         word_ids = model.generate(start_ids[-1])
#         word_ids = start_ids[:-1] + word_ids
#         txt = ''.join([id_to_word[i] for i in word_ids])
#         # txt = txt.replace('<eos>', '。\n')
#         txt = "。".join([txt, "\n"])
#         f.write(",".join([str(cnt), txt]))
#
#         print(cnt, start_sentence)
#         cnt += 1

start_sentence = '重磅会议后的四个暗示信号|市场观察'
start_words = np.array(word_sep_model.cut(start_sentence))[::, 0]
start_ids = [word_to_id[w] for w in start_words]


for x in start_ids[:-1]:
    x = np.array(x).reshape(1, 1)
    model.predict(x)

word_ids = model.generate(start_ids[-1])
word_ids = start_ids[:-1] + word_ids
txt = ''.join([id_to_word[i] for i in word_ids])
txt = txt.replace('<eos>', '。\n')

print(txt)
