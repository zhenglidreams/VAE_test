import sys
if '../' not in sys.path:
    sys.path.append('../')
import pandas as pd
from pathlib import Path
import random
import csv
import numpy as np
from settings import settings as stgs
from copy import deepcopy


n_pop = 2
min_depth = 3
max_depth = 9
window_pos = min_depth
max_window_size = 4
current_window_size = 1
tst_set = []
trn_set = []


def split_batch_file(dir_path, filename, test_pct = 0.2):
    full, tr, tst = [], [], []
    with open(dir_path/filename, 'r') as f:
        lines = f.readlines()
    for l in lines:  # ugly but it's only 2000 lines
        if l in tst_set or l in trn_set:
            if l in tst_set:
                tst.append(l)
            else:
                tr.append(l)
        else:
            if random.random() < test_pct:
                tst_set.append(l)
                tst.append(l)
            else:
                trn_set.append(l)
                tr.append(l)
    print(f'Split data into {len(tr)} training and {len(tst)} test samples.')
    with open(dir_path/'train.csv', 'w') as f:
        f.write(''.join(tr))
    with open(dir_path/'test.csv', 'w') as f:
        f.write(''.join(tst))


def split_population(path, filename, test_pct = 0.2):
    full_data = pd.read_csv(path/filename, names=['sentence', 'fitness'], index_col=False,
                                dtype={'sentences': str, 'fitness': np.float32})
    n_samples = full_data.shape[0]
    tr_idx = random.sample(list(range(n_samples)), int(n_samples * (1 - test_pct)))
    tst_idx = list(set(range(n_samples)).difference(tr_idx))
    print(f'Split data into {len(tr_idx)} training and {len(tst_idx)} test samples.')
    full_data.iloc[tr_idx, :].to_csv(stgs.PRED_BATCH_PATH/'train.csv')
    full_data.iloc[tst_idx, :].to_csv(stgs.PRED_BATCH_PATH/'test.csv')


def scrolling_gen_batch(dir_path, filename, win_size, win_pos, test_pct = 0.2):  # adds the batches in size window
    with open(dir_path/filename, 'r') as f:
        data = list(csv.reader(f))
    writer = csv.writer(
        open(dir_path/"windowbatch.csv", 'w', newline=''))
    node_sizes = list(range(win_pos, win_pos + win_size))
    print(f'Networks depths considered: {node_sizes}.')
    for row in data:
        cnt = row[0].count('/') - 1
        if cnt in node_sizes:
            writer.writerow(row)
    split_batch_file(stgs.PRED_BATCH_PATH, 'textfiles/windowbatch.csv', test_pct)


# stgs.WTS_PATH = Path('../../test_predictor/weights')
stgs.PRED_BATCH_PATH = Path('test_predictor/pred_batches')
# stgs.PRED_BATCH_PATH = Path('../../test_predictor/pred_batches')
stgs.PRED_HPARAMS["weights_path"] = Path('test_predictor/integrated_pred_ckpts')
# stgs.PRED_HPARAMS["weights_path"] = Path('../../test_predictor/pred_ckpts')
while window_pos + current_window_size - 1 <= max_depth:
    print('NEW BATCH SCROLL Window Size = ' + str(current_window_size) + ', Window Position = ' + str(window_pos))
    scrolling_gen_batch(stgs.PRED_BATCH_PATH/'textfiles', 'fitnessbatch.csv', current_window_size,
                        window_pos, test_pct=0.20
                        )
      