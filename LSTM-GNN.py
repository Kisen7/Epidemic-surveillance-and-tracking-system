class LSTM_GAT(nn.Module):
    def __init__(self, args):
        super(LSTM_GAT, self).__init__()
        self.args = args
        self.out_feats = 128
        self.gat = GAT(in_feats=args.hidden_size, h_feats=128, out_feats=64)
        self.lstm = nn.LSTM(input_size=args.input_size, hidden_size=args.hidden_size,
                            num_layers=args.num_layers, batch_first=True, dropout=0.5)
        self.fcs = nn.ModuleList()
        for k in range(args.input_size):
            self.fcs.append(nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, args.output_size)
            ))

    def create_edge_index(self, adj):
        adj = adj.cpu()
        ones = torch.ones_like(adj)
        zeros = torch.zeros_like(adj)
        edge_index = torch.where(adj > 0, ones, zeros)
        #
        edge_index_temp = sp.coo_matrix(edge_index.numpy())
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
        edge_index = torch.LongTensor(indices)
        # edge_weight
        edge_weight = []
        t = edge_index.numpy().tolist()
        for x, y in zip(t[0], t[1]):
            edge_weight.append(adj[x, y])
        edge_weight = torch.FloatTensor(edge_weight)
        edge_weight = edge_weight.unsqueeze(1)

        return edge_index.to(device), edge_weight.to(device)

    def forward(self, x):
        # x (b, s, i)
        x, _ = self.lstm(x)  # b, s, h
        # s * h conv
        s = torch.randn((x.shape[0], x.shape[1], 64)).to(device)
        for k in range(x.shape[0]):
            feat = x[k, :, :]  # s, h
            # creat edge_index
            adj = torch.matmul(feat, feat.T)  # s * s
            adj = F.softmax(adj, dim=1)
            edge_index, edge_weight = self.create_edge_index(adj)
            feat = self.gat(feat, edge_index, edge_weight)
            s[k, :, :] = feat

        # s(b, s, 64)
        s = s[:, -1, :]
        preds = []
        for fc in self.fcs:
            preds.append(fc(s))

        pred = torch.stack(preds, dim=0)

        return pred
