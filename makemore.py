import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import sys
import time

@dataclass
class ModelConfig:
    block_size: int = None # length of the input sequence of integers
    vocab_size: int = None
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    dropout: float = None

# ---------------------------------------------------------------------------------------------------
# Bigram LM

class Bigram(nn.Module):
    """
    Bigram Language Model: a lookup table built with counts of bigram pairs in the data
    """
    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros(n,n))
    
    def get_block_size(self):
        return 1 # only needs 1 previous char to predict next char

    def forward(self, idx, targets=None):
        # forward pass
        logits = self.logits[idx] # ()
    
        loss = None
        if targets is not None:
            # ignore -1 entries in the targets
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1), ignore_index = -1)
        return logits, loss


# ----------------------------------------------------------------------------------------------------
# Helper functions

@torch.no_grad()
def generate(model, idx, max_new_tokens):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t))
    and complete the sequence max_new_tokens times,
    feeding the predictions back into the model each time
    Most likely you'll want to make sure to be in model.eval() mode for this
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits,_ = model(idx_cond) # (b,t,vocab_size)
        logits = logits[:, -1, :] # (b,vocab_size)
        probs = F.softmax(logits, dim=-1) # (b,vocab_size)
        idx_next = torch.multinomial(probs, num_samples=1) # (b,1)
        idx = torch.cat((idx, idx_next), dim=1) # (b,t+1)
    return idx

def print_samples(num=10):
    """ samples from the model and print the decoded samples """
    # initialize X with 'num' rows of 0s
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1 # -1 coz we already start with <START> token
    X_samp = generate(model, X_init, max_new_tokens=steps)
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:] # cropped out the first <START> char
        
        # 0 is also the Stopping point, so getting the index of where 0 occurs, it it does
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)

        # seperately track samples that we have and have not seen before
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
        
        print('-'*80)
        for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
            print(f'{len(lst)} samples that are in {desc}')
            for word in lst:
                print(word)
        print('-'*80)


@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X,Y = batch
        _, loss = model(X,Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

class CharDataset(Dataset):
    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch: i+1 for i, ch in enumerate(chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words
    
    def get_vocab_size(self):
        return len(self.chars) + 1 # adding 1 for start token 0 or '.'
    
    def get_output_length(self):
        return self.max_word_length + 1 # +1 for <Start> token or '.'
    
    def encode(self, word):
        return torch.tensor([self.stoi[ix] for ix in word], dtype=torch.long)
    
    def decode(self, ix):
        # word = ''.join([self.itos[i] for i in ix])
        word = ''.join(self.itos[i] for i in ix) # more memory efficient; doesn't build full list before performing 'join'
        return word
    
    def __getitem__(self, idx):
        """
        returns: x, y each of length (max_word_length + 1)
        loads and returns a sample from the dataset
        """
        word = self.words[idx] # get the wore at idx index
        ix = self.encode(word) # get list of indexes for each char i the word
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long) # x of length = start_of_seq + max_word_length
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)

        #   word:     c   a   t
        #    ix:      3   1   20
        #    x:   [0, 3,  1,  20, 0, 0]
        #    y:   [3, 1, 20,  0, -1, -1]
        x[1: 1+len(ix)] = ix 
        y[:len(ix)]     = ix
        y[len(ix)+1:]   = -1

        return x, y

def create_datasets(input_file):
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words]
    words = [w for w in words if w] # get rid of any empty strings
    chars = sorted(list(set(''.join(words))))
    max_word_length = max(len(w) for w in words)
    print(f'Number of examples in the dataset: {len(words)}')
    print(f'max word length: {max_word_length}')
    print(f'number of unique characters in the vocab: {len(chars)}')
    print('Vocabulary: ')
    print(''.join(chars))

    # train / test set
    test_set_size = min(1000, int(len(words))*0.1)
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[: -test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size: ]]
    print(f'split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples')

    # wrap in dataset object
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset  = CharDataset(test_words , chars, max_word_length)

    return train_dataset, test_dataset

class InfiniteDataLoader:
    """
    Illusion of never ending to generate batches of data
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)
        """
        "iter":             is a built-in Python function
        "iter(obj)":        works for any obj for which __iter__() method is implemented
        "self.data_iter":   returns an iterator -
                            Which yields batches 1-by-1, when you call "next(self.data_iter)"
        """
    
    def next(self):
        try:
            batch = next(self.data_iter)
        
        # ⚠️ If a StopIteration exception happens here, don’t crash — just do something else instead.
        # if the iterator above is exhausted (after 1e10 samples)
        # we reset the iterator, and start getting "next" samples
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

# ------------------------------------------------------------------------------------------------------------------------
# The guard to avoid accidental execution when imported

if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description = "Lerning to build a HARNESS from JENKY jupyter notebooks")

    # system/input/output
    parser.add_argument('--input-file', '-i', type=str, default='names.txt', help='input file with things - one per line')
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--resume', action='store_true', help='when this flag is used, we will resume optimization form existing model in the work_dir')
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help='number of data workers for both train/test')
    parser.add_argument('--max_steps', type=int, default=-1, help='max number of optimization steps to run for, or -1 for infinite')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for compuyte, examples: cpu|cuda|cuda:2|mps')
    parser.add_argument('--seed', type=int, default=3407, help='seed')

    # sampling
    parser.add_argument('--top-k', type=int, default=-1, help='top-k for sampling, -1 means no top-k')

    # model
    parser.add_argument('--type', type=str, default='transformer', help='model class type to use bigram|mlp|rnn|gru|bow|transformer')
    parser.add_argument('--n-layer', type=int, default=4, help='number of layers')
    parser.add_argument('--n-head', type=int, default=4, help='number of heads in a transformer')
    parser.add_argument('--n-embd', type=int, default=64, help='number of feature channels in the model')
    parser.add_argument('--n-embd2', type=int, default=64, help='number of feature channels elsewhere in the model')

    # optimization
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='batch size during optimization')
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help='weight decay')

    args = parser.parse_args()
    print(vars(args))

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # init datasets
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length() # block_size = max_word_length + 1
    print(f'dataset determined that: {vocab_size=}, {block_size=}')

    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                         n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, n_embd2=args.n_embd2)
    if args.type == 'bigram':
        model = Bigram(config=config)
    else:
        raise ValueError(f'model type {args.type} is not recognized ')
    
    model.to(device=args.device)
    print(f'model #params: {sum(p.numel() for p in model.parameters())}')
    
    if args.resume or args.sample_only:
        print('resuming form existing model in the work_dir')
        model.load_state_dict(torch.load(os.path.join(args.word_dir, 'model.pt')))
    if args.sample_only:
        print_samples(num=50)
        sys.exit()

    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

    # init dataloader
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    # training loop
    best_loss = None
    step = 0
    while True:
        t0 = time.time()

        # get next batch, ship to device, unpach it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X,Y = batch

        # forward pass
        logits, loss = model(X,Y)

        # calculate gradients, update weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # waiting for all CUDA work on GPU to finish, then calculate iteration time taken
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 10 == 0:
            print(f'step: {step} | loss = {loss.item():.4f} | step time = {(t1-t0)*1000:.2f}ms')

        # evaluate the model
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=100)
            test_loss  = evaluate(model,  test_dataset, batch_size=100, max_batches=100)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()

            print(f'step {step} train loss: {train_loss} test loss: {test_loss}')

            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(f'test loss {test_loss} is the best so far, saving model to {out_path}')
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        # sample from the model
        if step > 0 and step % 200 == 0:
            print_samples(num=10)

        step += 1
        # termination condition
        if args.max_steps >= 0 and step >= args.max_steps:
            break

