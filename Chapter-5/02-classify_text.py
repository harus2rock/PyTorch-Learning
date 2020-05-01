import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import tqdm
import glob
import pathlib
import re
from statistics import mean

remove_marks_regex = re.compile('[,\.\(\)\[\]\*:;]|<.*?>')
shift_marks_regex = re.compile('([?!])')


def text2ids(text, vocab_dict):
    # remove marks except !?
    text = remove_marks_regex.sub('', text)
    # insert space between !? and a word
    text = shift_marks_regex.sub(r' \1 ', text)
    tokens = text.split()
    return [vocab_dict.get(token, 0) for token in tokens]


def list2tensor(token_idxes, max_len=100, padding=True):
    if len(token_idxes) > max_len:
        token_idxes = token_idxes[:max_len]
    n_tokens = len(token_idxes)
    if padding:
        token_idxes = token_idxes + [0] * (max_len - len(token_idxes))
    return torch.tensor(token_idxes, dtype=torch.int64), n_tokens


class IMDBDataset(Dataset):
    def __init__(self, dir_path, train=True, max_len=100, padding=True):
        self.max_len = max_len
        self.padding = padding

        path = pathlib.Path(dir_path)
        vocab_path = path.joinpath('imdb.vocab')

        # load vocav-file and split rows
        self.vocab_array = vocab_path.open().read().strip().splitlines()
        # make dict - key is a word, ID is the value
        self.vocab_dict = dict((w, i + 1) for (i, w) in enumerate(self.vocab_array))

        if train:
            target_path = path.joinpath('train')
        else:
            target_path = path.joinpath('test')
        pos_files = sorted(glob.glob(str(target_path.joinpath('pos/*.txt'))))
        neg_files = sorted(glob.glob(str(target_path.joinpath('neg/*.txt'))))
        # make tuple-list of (file_path, label): pos is 1, neg is 0
        self.labeled_files = \
            list(zip([0] * len(neg_files), neg_files)) + \
            list(zip([1] * len(pos_files), pos_files))

    @property
    def vocab_size(self):
        return len(self.vocab_array)

    def __len__(self):
        return len(self.labeled_files)

    def __getitem__(self, idx):
        label, f = self.labeled_files[idx]
        # convert to small letter
        data = open(f).read().lower()
        # convert text to id-list
        data = text2ids(data, self.vocab_dict)
        # convert id-list to tensor
        data, n_tokens = list2tensor(data, self.max_len, self.padding)
        return data, label, n_tokens


class SequenceTaggingNet(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=50,
                 hidden_size=50, num_layers=1, dropout=0.2):
        super().__init__()
        # padding_idx=0: Words with an ID of 0 will be converted to a vector of 0
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        # batch_first=True: input and output size are (batchsize, seqence, feature)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, h0=None, l=None):
        # convert IDs to multi-dim vector
        # x:(batch_size, step_size) -> (batch_size, step_size, embedding_dim)
        x = self.emb(x)
        # pass x and h0 to RNN
        # x:(batch_size, step_size, embedding_dim) -> (batch_size, step_size, hidden_dim)
        x, h = self.lstm(x, h0)
        # get only the last step
        # x:(batch_size, step_size, hidden_dim) -> (batch_size, 1)
        if l is not None:
            # 入力のもともとの長さがある場合はそれを使用する
            x = x[list(range(len(x))), l-1, :]
        else:
            # なければ単純に最後を利用する
            x = x[:, -1, :]
        # pass x to Linear
        x = self.linear(x)
        # x:(batch_size, 1) -> (batch_size,)
        x = x.squeeze()
        return x


def eval_net(net, data_loader, device='cpu'):
    net.eval()
    ys = []
    ypreds = []
    for x, y, l in data_loader:
        x = x.to(device)
        y = y.to(device)
        l = l.to(device)
        with torch.no_grad():
            y_pred = net(x, l=l)
            y_pred = (y_pred > 0).long()
            ys.append(y)
            ypreds.append(y_pred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    acc = (ys == ypreds).float().sum() / len(ys)
    return acc.item()


if __name__ == '__main__':
    base = pathlib.Path(__file__).parent
    train_data = IMDBDataset(base.joinpath('./data/aclImdb/'))
    test_data = IMDBDataset(base.joinpath('./data/aclImdb/'), train=False)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = SequenceTaggingNet(train_data.vocab_size + 1, num_layers=2)
    net.to(device)
    opt = optim.Adam(net.parameters())
    # Sigmoid Cross Entropy Loss
    loss_f = nn.BCEWithLogitsLoss()

    for epoch in range(10):
        losses = []
        net.train()
        for x, y, l in tqdm.tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)
            l = l.to(device)
            y_pred = net(x, l=l)
            loss = loss_f(y_pred, y.float())
            net.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_acc = eval_net(net, train_loader, device,)
        val_acc = eval_net(net, test_loader, device)
        print(epoch, mean(losses), train_acc, val_acc)
    # path = 'data/aclImdb/imdb.vocab'

    # # load vocav-file and split rows
    # vocab_array = path.open().read().strip().splitlines()
    # # make dict - key is a word, ID is the value
    # vocab_dict = dict((w, i + 1) for (i, w) in enumerate(vocab_array))

    # text = 'convert to small letter'

    # token_ids = text2ids(text, vocab_dict)
    # data, n_tokens = list2tensor(token_ids)

    # print(token_ids)
    # print(data)
    # print(n_tokens)
