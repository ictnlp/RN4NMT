from __future__ import division
import sys
import os
import re
import numpy
import shutil
import json
import subprocess
import math
import random

import torch as tc
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.append('../')
import wargs
from wargs import *
reload(sys)
sys.setdefaultencoding('utf-8')

def str1(content, encoding='utf-8'):
    return json.dumps(content, encoding=encoding, ensure_ascii=False, indent=4)
    pass

#DEBUG = True
DEBUG = False
#PAD = 0
#UNK = 1
#BOS = 2
#EOS = 3

PAD_WORD = '<pad>'
UNK_WORD = 'unk'
BOS_WORD = '<b>'
EOS_WORD = '<e>'

RESERVED_TOKENS = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD = RESERVED_TOKENS.index(PAD_WORD)  # Normally 0
# Normally None of 1
UNK = RESERVED_TOKENS.index(UNK_WORD) if UNK_WORD in RESERVED_TOKENS else None
BOS = RESERVED_TOKENS.index(BOS_WORD)  # Normally 1 or 2
EOS = RESERVED_TOKENS.index(EOS_WORD)  # Normally 2 or 3

epsilon = 1e-20

class XavierLinear(nn.Module):
    ''' Simple Linear layer with xavier init '''
    def __init__(self, d_in, d_out, bias=True):
        super(XavierLinear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal(self.linear.weight)
    def forward(self, x):
        return self.linear(x)

def _load_model(model_path):
    wlog('Loading pre-trained model ... from {} '.format(model_path), 0)
    state_dict = tc.load(model_path, map_location=lambda storage, loc: storage)
    if len(state_dict) == 4:
        model_dict, eid, bid, optim = state_dict['model'], state_dict['epoch'], state_dict['batch'], state_dict['optim']
        rst = ( model_dict, eid, bid, optim )
    elif len(state_dict) == 5:
        model_dict, class_dict, eid, bid, optim = state_dict['model'], state_dict['class'], state_dict['epoch'], state_dict['batch'], state_dict['optim']
        rst = ( model_dict, class_dict, eid, bid, optim )
    wlog('at epoch {} and batch {}'.format(eid, bid))
    wlog(optim)
    return rst

def toVar(x, isCuda=None, volatile=False):

    if not isinstance(x, tc.autograd.variable.Variable):
        if isinstance(x, int): x = tc.Tensor([x])
        elif isinstance(x, list): x = tc.Tensor(x)
        x = Variable(x, requires_grad=False, volatile=volatile)
        if isCuda is not None: x = x.cuda()

    return x

def clip(x, rate=0.):

    b1 = (x < 1 - rate).float()
    b2 = (x > 1 + rate).float()
    b3 = (((1 - rate <= x) + (x <= 1 + rate)) > 1).float()

    return b1 * (1 - rate) + b2 * (1 + rate) + b3 * x

def rm_elems_byid(l, ids):

    isTensor = isinstance(l, tc.FloatTensor)
    isTorchVar = isinstance(l, tc.autograd.variable.Variable)
    if isTensor is True: l = l.transpose(0, 1).tolist()
    if isTorchVar is True: l = l.transpose(0, 1).data.tolist() #  -> (B, srcL)

    if isinstance(ids, int): del l[ids]
    elif len(ids) == 1: del l[ids[0]]
    else:
        for idx in ids: l[idx] = PAD_WORD
        l = filter(lambda a: a != PAD_WORD, l)

    if isTensor is True: l = tc.Tensor(l).transpose(0, 1)  # -> (srcL, B')
    if isTorchVar is True:
        l = Variable(tc.Tensor(l).transpose(0, 1), requires_grad=False, volatile=True)
        if wargs.gpu_id: l = l.cuda()

    return l

# x, y are torch Tensors
def cor_coef(a, b, eps=1e-20):

    E_a, E_b = tc.mean(a), tc.mean(b)
    E_a_2, E_b_2 = tc.mean(a * a), tc.mean(b * b)
    rl_rho = tc.mean(a * b) - E_a * E_b
    #print 'a',rl_rho.data[0]
    D_a, D_b = E_a_2 - E_a * E_a, E_b_2 - E_b * E_b

    rl_rho = rl_rho / ( tc.sqrt(D_a * D_b) + eps )  # number stable
    del E_a, E_b, E_a_2, E_b_2, D_a, D_b

    return rl_rho

def format_time(time):
    '''
        :type time: float
        :param time: the number of seconds

        :print the text format of time
    '''
    rst = ''
    if time < 0.1: rst = '{:7.2f} ms'.format(time * 1000)
    elif time < 60: rst = '{:7.5f} sec'.format(time)
    elif time < 3600: rst = '{:6.4f} min'.format(time / 60.)
    else: rst = '{:6.4f} hr'.format(time / 3600.)

    return rst

def append_file(filename, content):

    f = open(filename, 'a')
    f.write(content + '\n')
    f.close()

def str_cat(pp, name):

    return '{}_{}'.format(pp, name)

def wlog(obj, newline=1):

    if newline == 1: sys.stderr.write('{}\n'.format(obj))
    else: sys.stderr.write('{}'.format(obj))

def debug(s, newline=1):

    if DEBUG is True:
        if newline == 1: sys.stderr.write('{}\n'.format(s))
        else: sys.stderr.write(s)
        sys.stderr.flush()

def get_gumbel(LB, V, eps=1e-30):

    return Variable(
        -tc.log(-tc.log(tc.Tensor(LB, V).uniform_(0, 1) + eps) + eps), requires_grad=False)

def BLToStrList(x, xs_L, return_list=False):

    x = x.data.tolist()
    B, xs = len(x), []
    for bidx in range(B):
        x_one = numpy.asarray(x[bidx][:int(xs_L[bidx])])
        #x_one = str(x_one.astype('S10'))[1:-1].replace('\n', '')
        x_one = str(x_one.astype('S10')).replace('\n', '')
        #x_one = x_one.__str__().replace('  ', ' ')[2:-1]
        xs.append(x_one)
    return xs if return_list is True else '\n'.join(xs)

def init_params(p, name='what', uniform=False):

    if uniform is True:
        p.data.uniform_(-0.1, 0.1)
        wlog('{:7} -> grad {}\t{}'.format('Uniform', p.requires_grad, name))
    else:
        if len(p.size()) == 2:
            if p.size(0) == 1 or p.size(1) == 1:
                p.data.zero_()
                wlog('{:7}-> grad {}\t{}'.format('Zero', p.requires_grad, name))
            else:
                p.data.normal_(0, 0.01)
                wlog('{:7}-> grad {}\t{}'.format('Normal', p.requires_grad, name))
        elif len(p.size()) == 1:
            p.data.zero_()
            wlog('{:7}-> grad {}\t{}'.format('Zero', p.requires_grad, name))

def init_dir(dir_name, delete=False):

    if not dir_name == '':
        if os.path.exists(dir_name):
            if delete:
                shutil.rmtree(dir_name)
                wlog('\n{} exists, delete'.format(dir_name))
            else:
                wlog('\n{} exists, no delete'.format(dir_name))
        else:
            os.mkdir(dir_name)
            wlog('\nCreate {}'.format(dir_name))

def part_sort(vec, num):
    '''
    vec:    [ 3,  4,  5, 12,  1,  3,  29999, 33,  2, 11,  0]
    '''

    idx = numpy.argpartition(vec, num)[:num]

    '''
    put k-min numbers before the _th position and get indexes of the k-min numbers in vec (unsorted)
    k-min vals:    [ 1,  0,  2, 3,  3,  ...]
    idx = np.argpartition(vec, 5)[:4]:
        [ 4, 10,  8,  0,  5]
    '''

    kmin_vals = vec[idx]

    '''
    kmin_vals:  [1, 0, 2, 3, 3]
    '''

    k_rank_ids = numpy.argsort(kmin_vals)

    '''
    k_rank_ids:    [1, 0, 2, 3, 4]
    '''

    k_rank_ids_invec = idx[k_rank_ids]

    '''
    k_rank_ids_invec:  [10,  4,  8,  0,  5]
    '''

    '''
    sorted_kmin = vec[k_rank_ids_invec]
    sorted_kmin:    [0, 1, 2, 3, 3]
    '''

    return k_rank_ids_invec

# beam search
def init_beam(beam, s0=None, cnt=50, score_0=0.0, loss_0=0.0, dyn_dec_tup=None, cp=False, transformer=False):
    del beam[:]
    for i in range(cnt + 1):
        ibeam = []  # one beam [] for one char besides start beam
        beam.append(ibeam)

    if cp is True:
        beam[0] = [ [ (loss_0, None, s0, 0, BOS, 0) ] ]
        return
    # indicator for the first target word (<b>)
    if dyn_dec_tup is not None:
        beam[0].append((loss_0, dyn_dec_tup, s0, BOS, 0))
    elif wargs.len_norm == 2:
        beam[0] = [[ (loss_0, None, s0[i], i, BOS, 0) ] for i in range(s0.size(0))]
    #elif with_batch == 0:
    #    beam[0].append((loss_0, s0, BOS, 0))
    else:
        beam[0] = [[ (loss_0, s0[i], i, BOS, 0) ] for i in range(s0.size(0))]

def back_tracking(beam, bidx, best_sample_endswith_eos, attent_probs=None):
    # (0.76025655120611191, [29999], 0, 7)
    best_loss, accum, _, w, bp, endi = best_sample_endswith_eos
    # starting from bp^{th} item in previous {end-1}_{th} beam of eos beam, w is <eos>
    seq = []
    attent_matrix = [] if attent_probs is not None else None
    check = (len(beam[0][0][0]) == 5)
    #print len(attent_probs), endi
    for i in reversed(xrange(0, endi)): # [0, endi-1], with <bos> 0 and no <eos> endi==self.maxL
        # the best (minimal sum) loss which is the first one in the last beam,
        # then use the back pointer to find the best path backward
        # <eos> is in pos endi, we do not keep <eos>
        if check is True:
            _, _, true_bidx, w, backptr = beam[i][bidx][bp]
            #if isinstance(true_bidx, int): assert true_bidx == bidx
        else: _, _, _, _, w, backptr = beam[i][bidx][bp]
        seq.append(w)
        bp = backptr
        # ([first word, ..., last word]) with bos and no eos, bos haven't align
        if attent_matrix is not None and i != 0:
            attent_matrix.append(attent_probs[i-1][:, bp])

    if attent_probs is not None and len(attent_matrix) > 0:
        # attent_matrix: (trgL, srcL)
        attent_matrix = tc.stack(attent_matrix[::-1], dim=0)
        attent_matrix = attent_matrix.cpu().data.numpy()

    # seq (bos, t1, t2, t3, t4, ---)
    # att (---, a0, a1, a2, a3, a4 ) 
    return seq[::-1], best_loss, attent_matrix # reverse

def filter_reidx(best_trans, tV_i2w=None, ifmv=False, ptv=None):

    if ifmv and ptv is not None:
        # OrderedDict([(0, 0), (1, 1), (3, 5), (8, 2), (10, 3), (100, 4)])
        # reverse: OrderedDict([(0, 0), (1, 1), (5, 3), (2, 8), (3, 10), (4, 100)])
        # part[index] get the real index in large target vocab firstly
        true_idx = [ptv[i] for i in best_trans]
    else:
        true_idx = best_trans

    true_idx = filter(lambda y: y != BOS and y != EOS, true_idx)

    return idx2sent(true_idx, tV_i2w), true_idx

def sent_filter(sent):

    list_filter = filter(lambda x: x != PAD and x!= BOS and x != EOS, sent)

    return list_filter

def idx2sent(vec, vcb_i2w):
    # vec: [int, int, ...]
    r = [vcb_i2w[idx] for idx in vec]
    sent = ' '.join(r)
    return sent

def dec_conf():

    wlog('\n######################### Construct Decoder #########################\n')
    wlog('# Naive beam search => ')

    wlog('\t Beam size: {}'
         '\n\t Vocab normalized: {}'
         '\n\t Length normalized: {}\n\n'.format(
             wargs.beam_size,
             True if wargs.vocab_norm else False,
             True if wargs.len_norm else False
         )
    )

def print_attention_text(attention_matrix, source_tokens, target_tokens, threshold=0.9, isP=False):
    """
    Return the alignment string from the attention matrix.
    Prints the attention matrix to standard out.
    :param attention_matrix: The attention matrix, np.ndarray, (trgL, srcL)
    :param source_tokens: A list of source tokens, List[str]
    :param target_tokens: A list of target tokens, List[str]
    :param threshold: The threshold for including an alignment link in the result, float
    """

    assert attention_matrix.shape[0] == len(target_tokens)

    if isP is True:
        wlog('  ', 0)
        for j in target_tokens: wlog('---', 0)
        wlog('')

    alnList = []
    src_max_ids, src_max_p = attention_matrix.argmax(1) + 1, attention_matrix.max(1)
    for (i, f_i) in enumerate(source_tokens):
        #maxJ, maxP = 0, 0.0

        if isP is True: wlog(' |', 0)
        for (j, _) in enumerate(target_tokens):
            align_prob = attention_matrix[j, i]
            if i == 0:  # start from 1
                alnList.append('{}:{}/{:.2f}'.format(src_max_ids[j], j+1, src_max_p[j]))
                #if maxP >= 0.5:
                #    alnList.append('{}:{}/{:.2f}'.format(i + 1, maxJ + 1, maxP))    # start from 1 here
            if isP is True:
                if align_prob > threshold: wlog('(*)', 0)
                elif align_prob > 0.4: wlog('(?)', 0)
                else: wlog('   ', 0)
            #if align_prob > maxP: maxJ, maxP = j, align_prob

        if isP is True: wlog(' | {}'.format(f_i))

    if isP is True:
        wlog('  ', 0)
        for j in target_tokens: wlog('---', 0)
        wlog('')
        for k in range(max(map(len, target_tokens))):
            wlog('  ', 0)
            for word in target_tokens:
                letter = word[k] if len(word) > k else ' '
                wlog(' {} '.format(letter), 0)
            wlog('')
        wlog('')

    return ' '.join(alnList)

def plot_attention(attention_matrix, source_tokens, target_tokens, filename):
    """
    Uses matplotlib for creating a visualization of the attention matrix.
    :param attention_matrix: The attention matrix, np.ndarray
    :param source_tokens: A list of source tokens, List[str]
    :param target_tokens: A list of target tokens, List[str]
    :param filename: The file to which the attention visualization will be written to, str
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pylab import mpl

    matplotlib.rc('font', family='sans-serif')
    matplotlib.rc('font', serif='HelveticaNeue')
    matplotlib.rc('font', serif='SimHei')
    #matplotlib.rc('font', serif='Microsoft YaHei')
    #mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    #mpl.rcParams['font.sans-serif'] = ['SimHei']
    #plt.rcParams['font.sans-serif']=['WenQuanYi Micro Hei']
    #mpl.rcParams['font.sans-serif']=['WenQuanYi Micro Hei']
    #plt.rcParams['font.sans-serif']=['WenQuanYi Micro Hei']
    #matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    #plt.rcParams['axes.unicode_minus'] = False
    #mpl.rcParams['axes.unicode_minus'] = False
    #zh_font = mpl.font_manager.FontProperties(fname='/home5/wen/miniconda2/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf')
    #zh_font = mpl.font_manager.FontProperties(
    #    fname='/home/wen/anaconda2/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/Microsoft Yahei.ttf')
    #en_font = mpl.font_manager.FontProperties(
    #    fname='/home/wen/anaconda2/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/Microsoft Yahei.ttf')
        #fname='/home/wen/anaconda2/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf')

    assert attention_matrix.shape[0] == len(target_tokens)

    plt.clf()
    #plt.imshow(attention_matrix.transpose(), interpolation="nearest", cmap="Greys")
    plt.imshow(attention_matrix, interpolation="nearest", cmap="Greys")
    #plt.xlabel("Source", fontsize=16)
    #plt.ylabel("Target", fontsize=16)

    #plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('top')
    #plt.xticks(fontsize=18, fontweight='bold')
    plt.xticks(fontsize=20)
    #plt.yticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=20)

    #plt.grid(True, which='minor', linestyle='-')
    #plt.gca().set_xticks([i for i in range(0, len(target_tokens))])
    #plt.gca().set_yticks([i for i in range(0, len(source_tokens))])
    plt.gca().set_xticks([i for i in range(0, len(source_tokens))])
    plt.gca().set_yticks([i for i in range(0, len(target_tokens))])
    #plt.gca().set_xticklabels(source_tokens, rotation='vertical')
    plt.gca().set_xticklabels(source_tokens, rotation=70, fontweight='bold')
    #plt.gca().set_xticklabels(source_tokens, rotation=70, fontweight='bold', FontProperties=zh_font)
    plt.gca().tick_params(axis='x', labelsize=20)
    #plt.gca().set_xticklabels(source_tokens, rotation=70, fontsize=20)

    #source_tokens = [unicode(k, "utf-8") for k in source_tokens]
    #plt.gca().set_yticklabels(source_tokens, rotation='horizontal', fontproperties=zh_font)
    plt.gca().set_yticklabels(target_tokens, fontsize=24, fontweight='bold')
    #plt.gca().set_yticklabels(target_tokens, fontsize=24, fontweight='bold', FontProperties=en_font)

    plt.tight_layout()
    #plt.draw()
    #plt.show()
    #plt.savefig(filename, format='png', dpi=400)
    #plt.grid(True)
    #plt.savefig(filename, dpi=400)
    plt.savefig(filename, format='svg', dpi=600, bbox_inches='tight')
    #plt.savefig(filename)
    wlog("Saved alignment visualization to " + filename)

def ids2Tensor(list_wids, bos_id=None, eos_id=None):
    # input: list of int for one sentence
    list_idx = [bos_id] if bos_id else []
    for wid in list_wids: list_idx.extend([wid])
    list_idx.extend([eos_id] if eos_id else [])
    return tc.LongTensor(list_idx)

def lp_cp(bp, beam_idx, bidx, beam):
    ys_pi = []
    assert len(beam[0][0][0]) == 6, 'require attention prob for alpha'
    for i in reversed(xrange(1, beam_idx)):
        _, p_im1, _, _, w, bp = beam[i][bidx][bp]
        ys_pi.append(p_im1)
    if len(ys_pi) == 0: return 1.0, 0.0
    ys_pi = tc.stack(ys_pi, dim=0).sum(0)   # (part_trg_len, src_len) -> (src_len, )
    m = ( ys_pi > 1.0 ).float()
    ys_pi = ys_pi * ( 1. - m ) + m
    lp = ( ( 5 + beam_idx - 1 ) ** wargs.alpha_len_norm ) / ( (5 + 1) ** wargs.alpha_len_norm )
    cp = wargs.beta_cover_penalty * ( ys_pi.log().sum().data[0] )

    return lp, cp

class MaskSoftmax(nn.Module):

    def __init__(self):

        super(MaskSoftmax, self).__init__()

    def forward(self, x, mask=None, dim=-1):

        # input torch tensor or variable, take max for numerical stability
        x_max = tc.max(x, dim=dim, keepdim=True)[0]
        x_minus = x - x_max
        x_exp = tc.exp(x_minus)
        if mask is not None: x_exp = x_exp * mask
        x = x_exp / ( tc.sum( x_exp, dim=dim, keepdim=True ) + epsilon )

        return x

class MyLogSoftmax(nn.Module):

    def __init__(self, self_norm_alpha=None):

        super(MyLogSoftmax, self).__init__()
        self.sna = self_norm_alpha

    def forward(self, x):

        # input torch tensor or variable
        x_max = tc.max(x, dim=-1, keepdim=True)[0]  # take max for numerical stability
        log_norm = tc.log( tc.sum( tc.exp( x - x_max ), dim=-1, keepdim=True ) + epsilon ) + x_max
        x = x - log_norm    # get log softmax

        # Sum_( log(P(xi)) - alpha * square( log(Z(xi)) ) )
        if self.sna is not None: x = x - self.sna * tc.pow(log_norm, 2)

        return log_norm, x

'''Layer normalize the tensor x, averaging over the last dimension.'''
class Layer_Norm(nn.Module):

    def __init__(self, d_hid, eps=1e-3):
        super(Layer_Norm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(tc.ones(d_hid), requires_grad=True)
        self.b = nn.Parameter(tc.zeros(d_hid), requires_grad=True)

    def forward(self, z):

        if z.size(-1) == 1: return z
        mu = tc.mean(z, dim=-1, keepdim=True)
        #sigma = tc.std(z, dim=-1, keepdim=True) # std has problems
        variance = tc.mean(tc.pow(z - mu, 2), dim=-1, keepdim=True)
        sigma = tc.sqrt(variance)
        z_norm = tc.div((z - mu), (sigma + self.eps))
        z_norm = z_norm * self.g.expand_as(z_norm) + self.b.expand_as(z_norm)
        return z_norm

from tools.bleu import *
def batch_search_oracle(B_hypos_list, y_LB, y_mask_LB):

    #print B_hypos_list
    #y_Ls contains the <s> and <e> of each sentence in one batch
    y_maxL, y_Ls = y_mask_LB.size(0), y_mask_LB.sum(0).data.int().tolist()
    #print y_maxL, y_Ls
    #for bidx, hypos_list in enumerate(B_hypos_list):
    #    for hypo in hypos_list:
    #        hypo += [PAD] * (y_maxL - y_Ls[bidx])
    #print B_hypos_list
    # B_hypos_list: [[[w0, w1, w2, ..., ], [w0, w1]], [sent0, sent1], [...]]
    oracles = []
    B_ys_list = BLToStrList(y_LB.t(), [l-1 for l in y_Ls], True) # remove <s> and <e>
    for bidx, (hyps, gold) in enumerate(zip(B_hypos_list, B_ys_list)):
        oracle, max_bleu = hyps[0], 0.
        for h in hyps:
            h_ = str(numpy.array(h[1:]).astype('S10')).replace('\n', '')    # remove bos
            # do not consider <s> and <e> when calculating BLEU
            assert len(h_.split(' ')) == len(gold.split(' '))
            BLEU = bleu(h_, [gold], logfun=debug)
            if BLEU > max_bleu:
                max_bleu = BLEU
                oracle = h
        oracles.append(oracle + [EOS] + [PAD] * (y_maxL - y_Ls[bidx]))
        # same with y_LB
    oracles = Variable(tc.Tensor(oracles).long().t(), requires_grad=False)

    return oracles

def grad_checker(model, _checks=None):

    _grad_nan = False
    for n, p in model.named_parameters():
        if p.grad is None:
            debug('grad None | {}'.format(n))
            continue
        tmp_grad = p.grad.data.cpu().numpy()
        if numpy.isnan(tmp_grad).any(): # we check gradient here for vanishing Gradient
            wlog("grad contains 'nan' | {}".format(n))
            #wlog("gradient\n{}".format(tmp_grad))
            _grad_nan = True
        if n == 'decoder.l_f1_0.weight' or n == 's_init.weight' or n=='decoder.l_f1_1.weight' \
           or n == 'decoder.l_conv.0.weight' or n == 'decoder.l_f2.weight':
            debug('grad zeros |{:5} {}'.format(str(not numpy.any(tmp_grad)), n))

    if _grad_nan is True and wargs.dynamic_cyk_decoding is True and _checks is not None:
        for _i, items in enumerate(_checks):
            wlog('step {} Variable----------------:'.format(_i))
            #for item in items: wlog(item.cpu().data.numpy())
            wlog('wen _check_tanh_sa ---------------')
            wlog(items[0].cpu().data.numpy())
            wlog('wen _check_a1_weight ---------------')
            wlog(items[1].cpu().data.numpy())
            wlog('wen _check_a1 ---------------')
            wlog(items[2].cpu().data.numpy())
            wlog('wen alpha_ij---------------')
            wlog(items[3].cpu().data.numpy())
            wlog('wen before_mask---------------')
            wlog(items[4].cpu().data.numpy())
            wlog('wen after_mask---------------')
            wlog(items[5].cpu().data.numpy())

def proc_bpe(input_fname, output_fname):

    fin = open(input_fname, 'r')
    contend = fin.read()
    fin.close()

    contend = re.sub('(@@ )|(@@ ?$)', '', contend)

    fout = open(output_fname, 'w')
    fout.write(contend)
    fout.close()

