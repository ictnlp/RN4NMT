# Maximal sequence length in training data
max_seq_len = 100

'''
Embedding layer
'''
# Size of word embedding of source word and target word
src_wemb_size = 512
trg_wemb_size = 512

'''
Encoder layer
'''
# Size of hidden units in encoder
enc_hid_size = 512

'''
Attention layer
'''
# Size of alignment vector
align_size = 512

'''
Decoder layer
'''
# Size of hidden units in decoder
dec_hid_size = 512
# Size of the output vector
out_size = 512
drop_rate = 0.5

# Directory to save model, test output and validation output
dir_model = 'wmodel'
dir_valid = 'wvalid'
dir_tests = 'wtests'

# Validation data
val_shuffle = True
# Training data
train_shuffle = True
batch_size = 80
sort_k_batches = 20

# Data path
dir_data = 'data/'
train_prefix = 'train'
train_src_suffix = 'src'
train_trg_suffix = 'trg'
dev_max_seq_len = 10000000

src_char = False    # whether split source side into characters
# Dictionary
src_dict_size = 30000
trg_dict_size = 30000
src_dict = dir_data + 'src.dict.tcf'
trg_dict = dir_data + 'trg.dict.tcf'

inputs_data = dir_data + 'inputs.pt'

with_bpe = False
with_postproc = False
# Training
max_epochs = 20
epoch_shuffle = False
epoch_shuffle_minibatch = 1

small = False
eval_small = False
epoch_eval = False
char = False

src_wemb_size = 256
trg_wemb_size = 256
enc_hid_size = 256
align_size = 256
dec_hid_size = 256
out_size = 256
val_tst_dir = './data/'
#val_tst_dir = '/home/wen/3.corpus/mt/nist_data_stanseg/'

val_prefix = 'devset1_2.lc'
val_src_suffix = 'zh'
val_ref_suffix = 'en'
tests_prefix = ['devset3.lc']
ref_cnt = 16
#val_prefix = 'nist02'
#val_src_suffix = 'src'
#val_ref_suffix = 'ref.plain_'
#tests_prefix = ['nist03', 'nist04', 'nist05', 'nist06', 'nist08', '900']
#ref_cnt = 4
batch_size = 40
max_epochs = 50
#src_dict_size = 32009
#trg_dict_size = 22822
epoch_eval = True
small = True
with_bpe = False
with_postproc = False
use_multi_bleu = False
cased = False

display_freq = 10 if small else 1000
sampling_freq = 100 if small else 5000
sample_size = 5
if_fixed_sampling = False
eval_valid_from = 500 if eval_small else 100000
eval_valid_freq = 100 if eval_small else 20000

save_one_model = True
start_epoch = 1

model_prefix = dir_model + '/model'
best_model = dir_valid + '/best.model.pt' if dir_valid else 'best.model.pt'
# pretrained model
pre_train = None
#pre_train = best_model
fix_pre_params = False

# decoder hype-parameters
beam_size = 10
vocab_norm = 1  # softmax
len_norm = 1    # 0: no noraml, 1: length normal, 2: alpha-beta
alpha_len_norm = 0.6
beta_cover_penalty = 0.

'''
Starting learning rate. If adagrad/adadelta/adam is used, then this is the global learning rate.
Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001
'''
opt_mode = 'adadelta'
learning_rate = 1.0
rho = 0.95

max_grad_norm = 1.0

# Start decaying every epoch after and including this epoch
start_decay_from = None
learning_rate_decay = 0.5
last_valid_bleu = 0.

snip_size = 10

# 1: rnnsearch, 4: relation network
model = 4

# convolutional layer
fltr_windows = [1, 3]
d_fltr_feats = [128, 256]
d_mlp = 256

print_att = False

gpu_id = [2]
#gpu_id = None

proj_share_weight=True
self_norm_alpha=None


