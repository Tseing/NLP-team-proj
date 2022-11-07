import sys
sys.path.append("..")
from common import config
# 用GPU运行时，打开下面的注释（cupy）
# ==============================================
# config.GPU = True
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ==============================================
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity, to_gpu
from better_rnnlm import BetterRnnlm
import preprocess


# 设定超参数
batch_size = 16
wordvec_size = 256
hidden_size = 256
time_size = 20
lr = 20.0
max_epoch = 5
max_grad = 0.25
dropout = 0.5

model_num = 10


# for model_idx in range(model_num):
    # 读入训练数据
corpus, word_to_id, id_to_word = preprocess.load_data("train")
corpus_val, _, _ = preprocess.load_data('val')

if config.GPU:
    corpus = to_gpu(corpus)
    corpus_val = to_gpu(corpus_val)

vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

model = BetterRnnlm(vocab_size, wordvec_size, hidden_size, dropout)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

best_ppl = float('inf')
for epoch in range(max_epoch):
    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size,
                time_size=time_size, max_grad=max_grad)

    model.reset_state()
    ppl = eval_perplexity(model, corpus_val)
    print("Model:", 'valid perplexity: ', ppl)

    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()
    else:
        lr /= 4.0
        optimizer.lr = lr

    model.reset_state()
    print('-' * 50)

print("Training Done!")
