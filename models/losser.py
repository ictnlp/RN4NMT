import torch as tc
import torch.nn as nn

import wargs
from tools.utils import *

class Classifier(nn.Module):

    def __init__(self, input_size, output_size, trg_lookup_table=None, trg_wemb_size=wargs.trg_wemb_size):

        super(Classifier, self).__init__()

        self.dropout = nn.Dropout(wargs.drop_rate)
        self.map_vocab = nn.Linear(input_size, output_size)

        if trg_lookup_table is not None:
            assert input_size == trg_wemb_size
            wlog('Copying weight of trg_lookup_table into classifier')
            self.map_vocab.weight = trg_lookup_table.weight
        self.log_prob = MyLogSoftmax(wargs.self_norm_alpha)

        weight = tc.ones(output_size)
        weight[PAD] = 0   # do not predict padding, same with ingore_index
        self.criterion = nn.NLLLoss(weight, size_average=False, ignore_index=PAD)

        self.output_size = output_size
        self.softmax = MaskSoftmax()

    def get_a(self, logit, noise=None):

        if not logit.dim() == 2: logit = logit.contiguous().view(-1, logit.size(-1))
        logit = self.map_vocab(logit)

        if noise is not None:
            logit.data.add_(
                -tc.log(-tc.log(
                    tc.Tensor(logit.size(0), logit.size(1)).cuda().uniform_(0, 1) + epsilon)
                    + epsilon)) / noise

        return logit

    def nll_loss(self, pred, gold, gold_mask):

        if pred.dim() == 3: pred = pred.view(-1, pred.size(-1))
        log_norm, pred = self.log_prob(pred)
        pred = pred * gold_mask[:, None]

        batch_Z = (log_norm * gold_mask[:, None]).abs().sum()

        return self.criterion(pred, gold), batch_Z

    def forward(self, feed, gold=None, gold_mask=None, noise=None):

        # no dropout in decoding
        feed = self.dropout(feed) if gold is not None else feed
        # (max_tlen_batch - 1, batch_size, out_size)
        pred = self.get_a(feed, noise)

        # decoding, if gold is None and gold_mask is None:
        if gold is None: return -self.log_prob(pred)[-1] if wargs.self_norm_alpha is None else -pred

        if gold.dim() == 2: gold, gold_mask = gold.view(-1), gold_mask.view(-1)
        # negative likelihood log
        nll, batch_Z = self.nll_loss(pred, gold, gold_mask)

        # (max_tlen_batch - 1, batch_size, trg_vocab_size)
        pred_correct = (pred.max(dim=-1)[1]).eq(gold).masked_select(gold.ne(PAD)).sum()

        # total loss,  correct count in one batch
        return nll, pred_correct, batch_Z

    #   outputs: the predict outputs from the model.
    #   gold: correct target sentences in current batch 
    def snip_back_prop(self, outputs, gold, gold_mask, shard_size=100):

        """
        Compute the loss in shards for efficiency.
        """
        batch_loss, batch_correct_num, batch_Z = 0, 0, 0
        cur_batch_count = outputs.size(1)
        shard_state = { "feed": outputs, "gold": gold, 'gold_mask': gold_mask }

        for shard in shards(shard_state, shard_size):
            loss, pred_correct, _batch_Z = self(**shard)
            batch_loss += loss.data.clone()[0]
            batch_correct_num += pred_correct.data.clone()[0]
            batch_Z += _batch_Z.data.clone()[0]
            loss.div(cur_batch_count).backward()

        return batch_loss, batch_correct_num, batch_Z

def filter_shard_state(state):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=True, volatile=False)
            yield k, v

def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute.make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.
    Yields:
        Each yielded shard is a dict.
    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield state
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, tc.split(v, shard_size))
                             for k, v in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            # each slice: return (('feed', 'gold', ...), (feed0, gold0, ...))
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
        inputs, grads = zip(*variables)
        tc.autograd.backward(inputs, grads)


