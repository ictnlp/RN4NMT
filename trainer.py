from __future__ import division

import os
import sys
import math
import time
import subprocess

import numpy as np
import torch as tc
from torch.autograd import Variable

import wargs
from tools.utils import *
from translate import Translator

class Trainer(object):

    def __init__(self, model, train_data, vocab_data, optim, valid_data=None, tests_data=None):

        self.model = model
        if isinstance(model, tc.nn.DataParallel): self.classifier = model.module.decoder.classifier
        else: self.classifier = model.decoder.classifier

        self.train_data = train_data
        self.sv = vocab_data['src'].idx2key
        self.tv = vocab_data['trg'].idx2key
        self.optim = optim
        self.valid_data = valid_data
        self.tests_data = tests_data

        self.model.train()

    def mt_eval(self, eid, bid):

        state_dict = { 'model': self.model.state_dict(), 'epoch': eid, 'batch': bid, 'optim': self.optim }

        if wargs.save_one_model: model_file = '{}.pt'.format(wargs.model_prefix)
        else: model_file = '{}_e{}_upd{}.pt'.format(wargs.model_prefix, eid, bid)
        tc.save(state_dict, model_file)
        wlog('Saving temporary model in {}'.format(model_file))

        self.model.eval()

        tor0 = Translator(self.model, self.sv, self.tv, print_att=wargs.print_att)
        bleu = tor0.trans_eval(self.valid_data, eid, bid, model_file, self.tests_data)

        self.model.train()

        return bleu

    def train(self):

        wlog('Start training ... ')
        assert wargs.sample_size < wargs.batch_size, 'Batch size < sample count'
        # [low, high)
        batch_count = len(self.train_data)
        batch_start_sample = tc.randperm(batch_count)[-1]
        wlog('Randomly select {} samples in the {}th/{} batch'.format(wargs.sample_size, batch_start_sample, batch_count))
        bidx, eval_cnt = 0, [0]
        wlog('Self-normalization alpha -> {}'.format(wargs.self_norm_alpha))
        tor_hook = Translator(self.model, self.sv, self.tv)

        train_start = time.time()
        wlog('\n' + '#' * 120 + '\n' + '#' * 30 + ' Start Training ' + '#' * 30 + '\n' + '#' * 120)

        _checks = None
        for epoch in range(wargs.start_epoch, wargs.max_epochs + 1):

            epoch_start = time.time()

            # train for one epoch on the training data
            wlog('\n' + '$' * 30, 0)
            wlog(' Epoch [{}/{}] '.format(epoch, wargs.max_epochs) + '$' * 30)
            if wargs.epoch_shuffle and epoch > wargs.epoch_shuffle_minibatch: self.train_data.shuffle()
            # shuffle the original batch
            shuffled_batch_idx = tc.randperm(batch_count)

            sample_size = wargs.sample_size
            epoch_loss, epoch_trg_words, epoch_num_correct, \
                    epoch_batch_logZ, epoch_n_sents = 0, 0, 0, 0, 0
            show_loss, show_src_words, show_trg_words, show_correct_num, \
                    show_batch_logZ, show_n_sents = 0, 0, 0, 0, 0, 0
            sample_spend, eval_spend, epoch_bidx = 0, 0, 0
            show_start = time.time()

            for k in range(batch_count):

                bidx += 1
                epoch_bidx = k + 1
                batch_idx = shuffled_batch_idx[k] if epoch >= wargs.epoch_shuffle_minibatch else k

                # (max_slen_batch, batch_size)
                _, srcs, spos, ttrgs_for_files, ttpos_for_files, slens, srcs_m, trg_mask_for_files = self.train_data[batch_idx]
                trgs, tpos, trgs_m = ttrgs_for_files[0], ttpos_for_files[0], trg_mask_for_files[0]
                self.optim.zero_grad()
                # (max_tlen_batch - 1, batch_size, out_size)
                gold, gold_mask = trgs[1:], trgs_m[1:]
                outputs = self.model(srcs, trgs[:-1], srcs_m, trgs_m[:-1])
                if len(outputs) == 2: (outputs, _checks) = outputs
                if len(outputs) == 2: (outputs, attends) = outputs

                this_bnum = outputs.size(1)
                epoch_n_sents += this_bnum
                show_n_sents += this_bnum
                #batch_loss, batch_correct_num, batch_log_norm = self.classifier(outputs, trgs[1:], trgs_m[1:])
                #batch_loss.div(this_bnum).backward()
                #batch_loss = batch_loss.data[0]
                #batch_correct_num = batch_correct_num.data[0]
                batch_loss, batch_correct_num, batch_Z = self.classifier.snip_back_prop(
                    outputs, gold, gold_mask, wargs.snip_size)

                self.optim.step()
                grad_checker(self.model, _checks)

                batch_src_words = srcs.data.ne(PAD).sum()
                assert batch_src_words == slens.data.sum()
                batch_trg_words = trgs[1:].data.ne(PAD).sum()

                show_loss += batch_loss
                show_correct_num += batch_correct_num
                epoch_loss += batch_loss
                epoch_num_correct += batch_correct_num
                show_src_words += batch_src_words
                show_trg_words += batch_trg_words
                epoch_trg_words += batch_trg_words

                show_batch_logZ += batch_Z
                epoch_batch_logZ += batch_Z

                if epoch_bidx % wargs.display_freq == 0:
                    #print show_correct_num, show_loss, show_trg_words, show_loss/show_trg_words
                    ud = time.time() - show_start - sample_spend - eval_spend
                    wlog(
                        'Epo:{:>2}/{:>2} |[{:^5} {:^5} {:^5}k] |acc:{:5.2f}% |ppl:{:4.2f} '
                        '||w-logZ|:{:.2f} ||s-logZ|:{:.2f} '
                        '|stok/s:{:>4}/{:>2}={:>2} |ttok/s:{:>2} '
                        '|stok/sec:{:6.2f} |ttok/sec:{:6.2f} |lr:{:7.6f} |elapsed:{:4.2f}/{:4.2f}m'.format(
                            epoch, wargs.max_epochs, epoch_bidx, batch_idx, bidx/1000,
                            (show_correct_num / show_trg_words) * 100,
                            math.exp(show_loss / show_trg_words), show_batch_logZ / show_trg_words,
                            show_batch_logZ / show_n_sents,
                            batch_src_words, this_bnum, int(batch_src_words / this_bnum),
                            int(batch_trg_words / this_bnum),
                            show_src_words / ud, show_trg_words / ud, self.optim.learning_rate, ud,
                            (time.time() - train_start) / 60.)
                    )
                    show_loss, show_src_words, show_trg_words, show_correct_num, \
                            show_batch_logZ, show_n_sents = 0, 0, 0, 0, 0, 0
                    sample_spend, eval_spend = 0, 0
                    show_start = time.time()

                if epoch_bidx % wargs.sampling_freq == 0:

                    sample_start = time.time()
                    self.model.eval()
                    # (max_len_batch, batch_size)
                    sample_src_tensor = srcs.t()[:sample_size]
                    sample_trg_tensor = trgs.t()[:sample_size]
                    tor_hook.trans_samples(sample_src_tensor, sample_trg_tensor)
                    wlog('')
                    sample_spend = time.time() - sample_start
                    self.model.train()

                # Just watch the translation of some source sentences in training data
                if wargs.if_fixed_sampling and bidx == batch_start_sample:
                    # randomly select sample_size sample from current batch
                    rand_rows = np.random.choice(this_bnum, sample_size, replace=False)
                    sample_src_tensor = tc.Tensor(sample_size, srcs.size(0)).long()
                    sample_src_tensor.fill_(PAD)
                    sample_trg_tensor = tc.Tensor(sample_size, trgs.size(0)).long()
                    sample_trg_tensor.fill_(PAD)

                    for id in xrange(sample_size):
                        sample_src_tensor[id, :] = srcs.t()[rand_rows[id], :]
                        sample_trg_tensor[id, :] = trgs.t()[rand_rows[id], :]

                if wargs.epoch_eval is not True and bidx > wargs.eval_valid_from and \
                   bidx % wargs.eval_valid_freq == 0:

                    eval_start = time.time()
                    eval_cnt[0] += 1
                    wlog('\nAmong epoch, batch [{}], [{}] eval save model ...'.format(
                        epoch_bidx, eval_cnt[0]))

                    self.mt_eval(epoch, epoch_bidx)

                    eval_spend = time.time() - eval_start

            avg_epoch_loss = epoch_loss / epoch_trg_words
            avg_epoch_acc = epoch_num_correct / epoch_trg_words
            wlog('\nEnd epoch [{}]'.format(epoch))
            wlog('Train accuracy {:4.2f}%'.format(avg_epoch_acc * 100))
            wlog('Average loss {:4.2f}'.format(avg_epoch_loss))
            wlog('Train perplexity: {0:4.2f}'.format(math.exp(avg_epoch_loss)))
            wlog('Train average |w-logZ|: {}/{}={} |s-logZ|: {}/{}={}'.format(
                epoch_batch_logZ, epoch_trg_words, epoch_batch_logZ / epoch_trg_words,
                epoch_batch_logZ, epoch_n_sents, epoch_batch_logZ / epoch_n_sents))
            wlog('End epoch, batch [{}], [{}] eval save model ...'.format(epoch_bidx, eval_cnt[0]))

            mteval_bleu = self.mt_eval(epoch, epoch_bidx)
            # decay the probability value epslion of scheduled sampling per batch
            self.optim.update_learning_rate(mteval_bleu, epoch)
            epoch_time_consume = time.time() - epoch_start
            wlog('Consuming: {:4.2f}s'.format(epoch_time_consume))

        wlog('Finish training, comsuming {:6.2f} hours'.format((time.time() - train_start) / 3600))
        wlog('Congratulations!')

