from __future__ import division

import math
import wargs
import torch as tc
from utils import *
from torch.autograd import Variable

class Input(object):

    def __init__(self, src_tlst, trg_tlst, batch_size, volatile=False,
                 batch_sort=False, prefix=None, printlog=True):

        assert isinstance(src_tlst[0], tc.LongTensor), 'Require only one file in source side.'
        self.src_tlst = src_tlst
        cnt_sent = len(src_tlst)

        if trg_tlst is not None:
            self.trgs_tlst_for_files = trg_tlst
            assert isinstance(trg_tlst[0], list), 'Require file >=1 in target side.'
            # [sent0:[ref0, ref1, ...], sent1:[ref0, ref1, ... ], ...]
            assert cnt_sent == len(trg_tlst)
            if printlog is True:
                wlog('Build bilingual Input, Batch size {}, Sort in batch? {}'.format(batch_size, batch_sort))
            else:
                debug('Build bilingual Input, Batch size {}, Sort in batch? {}'.format(batch_size, batch_sort))
        else:
            self.trgs_tlst_for_files = None
            wlog('Build monolingual Input, Batch size {}, Sort in batch? {}'.format(batch_size, batch_sort))

        self.batch_size = batch_size
        self.gpu_id = wargs.gpu_id
        self.volatile = volatile
        self.batch_sort = batch_sort

        self.num_of_batches = int(math.ceil(cnt_sent / self.batch_size))
        self.prefix = prefix    # the prefix of data file, such as 'nist02' or 'nist03'

    def __len__(self):

        return self.num_of_batches
        #return len(self.src_tlst)

    def handle_batch(self, batch, right_align=False):

        multi_files = True if isinstance(batch[0], list) else False
        if multi_files is True:
            # [sent_0:[ref0, ref1, ...], sent_1:[ref0, ref1, ... ], ...]
            # -> [ref_0:[sent_0, sent_1, ...], ref_1:[sent_0, sent_1, ... ], ...]
            batch_for_files = [[one_sent_refs[ref_idx] for one_sent_refs in batch] \
                    for ref_idx in range(len(batch[0]))]
        else:
            # [src0, src1, ...] -> [ [src0, src1, ...] ]
            batch_for_files = [batch]

        pad_batch_for_files, lens_for_files, pos_pad_batch_for_files = [], [], []
        for batch in batch_for_files:   # a batch for one source/target file
            lens = [ts.size(0) for ts in batch]
            self.this_batch_size = len(batch)
            max_len_batch = max(lens)

            # (B, L)
            pad_batch = tc.Tensor(self.this_batch_size, max_len_batch).long()
            pad_batch.fill_(PAD)

            pos_pad_batch = tc.Tensor(self.this_batch_size, max_len_batch).long()
            pos_pad_batch.fill_(PAD)

            for idx in range(self.this_batch_size):
                length = lens[idx]
                offset = max_len_batch - length if right_align else 0
                # modify Tensor pad_batch
                pad_batch[idx].narrow(0, offset, length).copy_(batch[idx])
                pos_pad_batch[idx].narrow(0, offset, length).copy_(tc.arange(1, length+1))

            pad_batch_for_files.append(pad_batch)
            lens_for_files.append(lens)
            pos_pad_batch_for_files.append(pos_pad_batch)

        return pad_batch_for_files, lens_for_files, pos_pad_batch_for_files

    def __getitem__(self, idx):

        assert idx < self.num_of_batches, \
                'idx:{} >= number of batches:{}'.format(idx, self.num_of_batches)

        src_batch = self.src_tlst[idx * self.batch_size : (idx + 1) * self.batch_size]

        srcs, slens, spos_files = self.handle_batch(src_batch)
        assert len(srcs) == 1, 'Requires only one in source side.'
        srcs, slens, spos = srcs[0], slens[0], spos_files[0]

        if self.trgs_tlst_for_files is not None:
            # [sent_0:[ref0, ref1, ...], sent_1:[ref0, ref1, ... ], ...]
            trg_batch = self.trgs_tlst_for_files[idx * self.batch_size : (idx + 1) * self.batch_size]
            trgs_for_files, tlens_for_files, tpos_for_files = self.handle_batch(trg_batch)
            # -> [ref_0:[sent_0, sent_1, ...], ref_1:[sent_0, sent_1, ... ], ...]

        # sort the source and target sentence
        idxs = range(self.this_batch_size)

        if self.batch_sort is True:
            if self.trgs_tlst_for_files is None:
                zipb = zip(idxs, srcs, spos, slens)
                idxs, srcs, spos, slens = zip(*sorted(zipb, key=lambda x: x[-1]))
            else:
                #assert len(trgs_for_files) == 1, 'Unsupport to sort validation in one batch.'
                #zipb = zip(idxs, srcs, trgs_for_files[0], slens)
                #zipb = zip(idxs, srcs, tc.stack(trgs_for_files).permute(1, 0, 2), slens)
                # max length in different refs may differ, so can not tc.stack
                zipb = zip(idxs, srcs, spos, zip(*trgs_for_files), zip(*tpos_for_files), slens)
                idxs, srcs, spos, trgs, tpos, slens = zip(*sorted(zipb, key=lambda x: x[-1]))
                #trgs_for_files = [trgs]
                #trgs_for_files = list(tc.stack(trgs).permute(1, 0, 2))
                trgs_for_files = [tc.stack(ref) for ref in zip(*list(trgs))]
                tpos_for_files = [tc.stack(pos) for pos in zip(*list(tpos))]

        lengths = tc.IntTensor(slens).view(1, -1)   # (1, batch_size)
        lengths = Variable(lengths, volatile=self.volatile)

        def tuple2Tenser(x):

            if x is None: return x
            # (batch_size, max_len_batch) -> (max_len_batch, batch_size)
            x = tc.stack(x, dim=0).t().contiguous()
            if wargs.gpu_id: x = x.cuda()    # push into GPU

            return Variable(x, volatile=self.volatile)

        tsrcs, tspos = tuple2Tenser(srcs), tuple2Tenser(spos)
        src_mask = tsrcs.ne(0).float()

        if self.trgs_tlst_for_files is not None:

            ttrgs_for_files = [tuple2Tenser(trgs) for trgs in trgs_for_files]
            trg_mask_for_files = [ttrgs.ne(0).float() for ttrgs in ttrgs_for_files]
            ttpos_for_files = [tuple2Tenser(tpos) for tpos in tpos_for_files]

            '''
                [list] idxs: sorted idx by ascending order of source lengths in one batch
                [Variable] tsrcs: padded source batch, Variable (max_len_batch, batch_size)
                [Variable] tspos: padded source position batch, Variable (max_len_batch, batch_size)
                [list] ttrgs_for_files: list of Variables (padded target batch),
                            [Variable (max_len_batch, batch_size), ..., ]
                            each item in this list for one target reference file one batch
                [list] ttpos_for_files: list of Variables (padded target position batch),
                            [Variable (max_len_batch, batch_size), ..., ]
                            each item in this list for one target reference file one batch
                [intTensor] lengths: sorted source lengths by ascending order, (1, batch_size)
                [Variable] src_mask: 0/1 Variable (0 for padding) (max_len_batch, batch_size)
                [list] trg_mask_for_files: list of 0/1 Variables (0 for padding)
                            [Variable (max_len_batch, batch_size), ..., ]
                            each item in this list for one target reference file one batch
            '''
            return idxs, tsrcs, tspos, ttrgs_for_files, ttpos_for_files, lengths, src_mask, trg_mask_for_files

        else:

            return idxs, tsrcs, tspos, lengths, src_mask


    def shuffle(self):

        assert len(self.trgs_tlst_for_files) == 1, 'Unsupport to shuffle the whole validation set.'
        data = list(zip(self.src_tlst, self.trg_tlst))
        self.src_tlst, self.trg_tlst = zip(*[data[i] for i in tc.randperm(len(data))])




