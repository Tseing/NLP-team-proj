# -*- coding: UTF-8 -*-
import preprocess


corpus, word_to_id, id_to_word = preprocess.load_data('train', is_multi_file=True, file_idx=0)
