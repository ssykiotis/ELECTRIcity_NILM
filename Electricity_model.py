from model_helpers import *



class TransformerModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args

        self.original_len = args.window_size
        self.latent_len   = int(self.original_len / 2)
        self.dropout_rate = args.drop_out

        self.hidden       = args.hidden
        self.heads        = args.heads
        self.n_layers     = args.n_layers
        self.output_size  = args.output_size

        self.conv = nn.Conv1d(in_channels=1, out_channels=self.hidden, kernel_size=5, stride=1, padding=2, padding_mode='replicate')
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)

        self.position   = PositionalEmbedding(max_len=self.latent_len, d_model=self.hidden)
        self.layer_norm = LayerNorm(self.hidden)
        self.dropout    = nn.Dropout(p=self.dropout_rate)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(self.hidden, self.heads, self.hidden * 4, self.dropout_rate) for _ in range(self.n_layers)])

        self.deconv  = nn.ConvTranspose1d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=4, stride=2, padding=1)
        self.linear1 = nn.Linear(self.hidden, 128)
        self.linear2 = nn.Linear(128, self.output_size)

        self.truncated_normal_init()

    def truncated_normal_init(self, mean = 0, std = 0.02, lower = -0.04, upper = 0.04):
        params = list(self.named_parameters())
        for n, p in params:
            if 'layer_norm' in n:
                continue
            else:
                with torch.no_grad():
                    l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
                    u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)
    
    def forward(self, sequence):
        x_token   = self.pool(self.conv(sequence.unsqueeze(1))).permute(0, 2, 1)
        embedding = x_token + self.position(sequence)
        x = self.dropout(self.layer_norm(embedding))

        mask = None
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        x = self.deconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        return x

#TODO: unsqueeze to add dimension at 1



class ELECTRICITY(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.Discriminator = TransformerModel(args)
        self.pretrain      = args.pretrain
        args_gen           = args
        args_gen.hidden    = 64
        self.Generator     = TransformerModel(args_gen)

    def forward(self,sequence,mask=None):
        if self.pretrain:
            gen_out = self.Generator(sequence).squeeze()
            disc_in = sequence
            disc_in[mask] = gen_out[mask]
        else:
            disc_in = sequence
        disc_out = self.Discriminator(disc_in).squeeze()
        if self.pretrain:
            return disc_out, gen_out
        else:
            return disc_out, None