import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys 
sys.path.append('..')
from DataLoader import get_Yp_modes

class Y2mode(nn.Module):
    def __init__(self, in_dim = 1024, h_dim = [512, 256, 64], out_dim = 3):
        super().__init__()
        self.y2h = nn.Sequential(
            nn.Linear(in_dim, h_dim[0]),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim[0])
        )
        hidden_layers = []
        for i in range(len(h_dim) - 1):
            hidden_layers.extend(
                [
                nn.Linear(h_dim[i], h_dim[i+1]),
                nn.ReLU(), 
                nn.BatchNorm1d(h_dim[i+1])
                ]
            )
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.h2mode = nn.Sequential(
            nn.Linear(h_dim[-1], out_dim)
        )
    
    def forward(self, y):
        h = self.y2h(y)
        for layer in self.hidden_layers:
            h = layer(h)
        out = self.h2mode(h)
        return out


def main():
    model = Y2mode()
    device = 'cuda'
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)
    
    train_set, val_set = get_Yp_modes()
    train_dataloader = DataLoader(
        dataset=train_set, 
        batch_size=128,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        pin_memory=True)
    val_dataloader = DataLoader(
        dataset=val_set, 
        batch_size=128,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )
    best_acc = 0.0
    for epoch in range(20):
        it = 0
        model.eval()
        preds = []
        modes = []
        for (Yp, mode) in val_dataloader:
            Yp = Yp.float().to(device)
            mode = mode.to(device)
            pred = model(Yp)
            pred = torch.argmax(pred, dim = 1, keepdim=False)
            preds.append(pred)
            modes.append(mode)
        preds = torch.cat(preds, 0)
        modes = torch.cat(modes, 0)
        acc = (preds == modes).float().mean()
        print('epoch: {} || Acc: {}'.format(epoch, acc))
        if acc > best_acc:
            torch.save(model.state_dict(), './best_ckpt.pth')


        model.train()
        for (Yp, mode) in train_dataloader:
            it += 1
            Yp = Yp.float().to(device)
            label = mode.to(device)
            pred = model(Yp).float()

            loss = nn.CrossEntropyLoss()(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(pred, 1, keepdim=False)
            acc = (pred == label).float().mean()
            if it % 100 == 0:
                print('iter: {} || loss: {}, acc: {}'.format(it, loss.item(), acc.item()))
        

if __name__ == "__main__":
    main()