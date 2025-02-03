from model import *
import torch.nn.functional as F
F.silu()
class Encoder(nn.Module):
    def __init__(self, args:ModelArgs):
        super(Encoder, self).__init__()
        
        self.embedding = TokenEmbedding(args)
        self.layers = nn.ModuleList(
            [EncoderLayer(args) for _ in range(args.n_layer)]
        )

    def forward(self, x, mask):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, args:ModelArgs):
        super(Decoder, self).__init__()

        self.embedding = TokenEmbedding(args)
        self.layers = nn.ModuleList(
            [DecoderLayer(args) for _ in range(args.n_layer)]
        )
        self.fc = nn.Linear(args.dim, args.dec_voc_size)
    def forward(self, enc, dec, padding_mask, mask):
        dec = self.embedding(dec)

        for layer in self.layers:
            x = layer(dec, enc, padding_mask, mask)
        return x
    
class Transformer(nn.Module):
    def __init__(self, args:ModelArgs, device):
        super(Transformer, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.src_pad_idx = args.src_pad_idx
        self.trg_pad_idx = args.trg_pad_idx

    def make_casual_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones((len_q, len_k)).type(torch.BoolTensor).to(self.device))
        return mask

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)

        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)

        mask = q & k
        return mask

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * self.make_casual_mask(trg, trg)
        # trg ： decoder layer的输入
        # src :  encoder layer的输入
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        enc = self.encoder(src, src_mask)
        output = self.decoder(trg, enc, trg_mask, src_trg_mask) 
        return output



        # (batch, Time, len_q, len_k)



