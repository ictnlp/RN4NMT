#( -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
import time
import argparse
import subprocess
import torch as tc
from torch import cuda

import wargs
from tools.inputs import Input
from tools.utils import _load_model

from translate import Translator
from inputs_handler import extract_vocab, wrap_tst_data, wrap_data
from models.losser import *

if __name__ == "__main__":

    A = argparse.ArgumentParser(prog='NMT translator ... ')

    A.add_argument("-m", '--model-file', required=True, dest='model_file', help='model file')
    A.add_argument("-i", '--input-file', dest='input_file', default=None,
                   help='name of file to be translated')
    args = A.parse_args()

    model_file = args.model_file

    wlog('Using model: {}'.format(wargs.model))

    if wargs.model == 1: from models.rnnsearch import *
    elif wargs.model == 4: from models.rnnsearch_rn import *

    assert os.path.exists(wargs.src_dict) and os.path.exists(wargs.trg_dict), 'need vocabulary ...'
    src_vocab = extract_vocab(None, wargs.src_dict)
    trg_vocab = extract_vocab(None, wargs.trg_dict)

    src_vocab_size, trg_vocab_size = src_vocab.size(), trg_vocab.size()
    wlog('Vocabulary size: |source|={}, |target|={}'.format(src_vocab_size, trg_vocab_size))

    wlog('Start decoding ... init model ... ', 0)

    nmtModel = NMT(src_vocab_size, trg_vocab_size)

    if wargs.gpu_id:
        cuda.set_device(wargs.gpu_id[0])
        wlog('Push model onto GPU {} ... '.format(wargs.gpu_id[0]), 0)
        nmtModel.cuda()
    else:
        wlog('Push model onto CPU ... ', 0)
        nmtModel.cpu()
    wlog('done.')

    _dict = _load_model(model_file)
    if len(_dict) == 4: model_dict, eid, bid, optim = _dict
    elif len(_dict) == 5: model_dict, class_dict, eid, bid, optim = _dict

    nmtModel.load_state_dict(model_dict)
    wlog('\nFinish to load model.')

    dec_conf()

    nmtModel.eval()
    tor = Translator(nmtModel, src_vocab.idx2key, trg_vocab.idx2key, print_att=wargs.print_att)

    if not args.input_file:
        wlog('Translating one sentence ... ')
        s = '新奥尔良 是 爵士 音乐 的 发源 地 。'
        t = "When Lincoln goes to New Orleans, I hear Mississippi river's singing sound"
        s = [src_vocab.key2idx[x] if x in src_vocab.key2idx else UNK for x in s.split(' ')]
        s = tc.Tensor([s])
        t = [trg_vocab.key2idx[x] if x in trg_vocab.key2idx else UNK for x in t.split(' ')]
        t = tc.Tensor([t])
        pv = tc.Tensor([0, 10782, 2102, 1735, 4, 1829, 1657, 29999])
        tor.trans_samples(s, t)
        sys.exit(0)

    input_file = '{}{}.{}'.format(wargs.val_tst_dir, args.input_file, wargs.val_src_suffix)
    ref_file = '{}{}.{}'.format(wargs.val_tst_dir, args.input_file, wargs.val_ref_suffix)

    wlog('Translating test file {} ... '.format(input_file))
    test_src_tlst, test_src_lens = wrap_tst_data(input_file, src_vocab)
    test_input_data = Input(test_src_tlst, None, 1, volatile=True)

    batch_tst_data = None
    if os.path.exists(ref_file):
        wlog('With force decoding test file {} ... to get alignments'.format(input_file))
        wlog('\t\tRef file {}'.format(ref_file))
        tst_src_tlst, tst_trg_tlst = wrap_data(wargs.val_tst_dir, args.input_file,
                                               wargs.val_src_suffix, wargs.val_ref_suffix,
                                               src_vocab, trg_vocab, False, False, 1000000)
        batch_tst_data = Input(tst_src_tlst, tst_trg_tlst, 1, batch_sort=False)

    trans, alns = tor.single_trans_file(test_input_data, batch_tst_data=batch_tst_data)

    p1 = 'nbs'
    p2 = 'GPU' if wargs.gpu_id else 'CPU'

    #input_file_name = input_file if '/' not in input_file else input_file.split('/')[-1]
    outdir = 'wexp-{}-{}-{}-{}'.format(args.input_file, p1, p2, model_file.split('/')[0])
    init_dir(outdir)
    outprefix = outdir + '/trans_' + args.input_file
    # wTrans/trans
    file_out = '{}_e{}_upd{}_b{}'.format(outprefix, eid, bid, wargs.beam_size)

    mteval_bleu = tor.write_file_eval(file_out, trans, args.input_file, alns)

    bleus_record_fname = '{}/record_bleu.log'.format(outdir)
    bleu_content = 'epoch [{}], batch[{}], BLEU score : {}'.format(eid, bid, mteval_bleu)
    with open(bleus_record_fname, 'a') as f:
        f.write(bleu_content + '\n')
        f.close()

    sfig = '{}.{}'.format(outprefix, 'sfig')
    sfig_content = ('{} {} {} {}').format(
        eid,
        bid,
        wargs.beam_size,
        mteval_bleu
    )
    append_file(sfig, sfig_content)


