"""word_spectrum_generator.py —— 词级 64d向量→短频谱 MLP (80×16帧)"""
import numpy as np, torch, torch.nn as nn, os
from torch.utils.data import DataLoader, TensorDataset

class WordSpectrumGenerator(nn.Module):
    def __init__(self, vec_dim=64, mel_bins=80, n_frames=16, hidden=128):
        super().__init__()
        self.mel_bins, self.n_frames = mel_bins, n_frames
        self.net = nn.Sequential(
            nn.Linear(vec_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden*2), nn.ReLU(),
            nn.Linear(hidden*2, mel_bins*n_frames), nn.Sigmoid())
    def forward(self, x):
        return self.net(x).view(-1, self.mel_bins, self.n_frames)

def train(data_path=None, epochs=500, batch_size=32, lr=1e-3):
    if data_path is None: data_path = os.path.join(os.path.dirname(__file__), 'word_spectrum_dataset.npy')
    raw = np.load(data_path, allow_pickle=True)
    vecs = np.stack([r[0] for r in raw]); mels = np.stack([r[1] for r in raw])
    words = [r[2] for r in raw]
    print(f"Dataset: {len(vecs)} words, vec={vecs.shape}, mel={mels.shape}")

    v_mean, v_std = vecs.mean(0,keepdims=True), vecs.std(0,keepdims=True)+1e-8
    vecs_n = (vecs-v_mean)/v_std
    mel_min, mel_max = mels.min(), mels.max()
    mels_n = (mels-mel_min)/(mel_max-mel_min+1e-8)

    loader = DataLoader(TensorDataset(torch.from_numpy(vecs_n).float(), torch.from_numpy(mels_n).float()), batch_size=batch_size, shuffle=True)
    model = WordSpectrumGenerator()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        total = 0
        for bv, bm in loader: pred=model(bv); loss=loss_fn(pred,bm); opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()
        if (ep+1)%50==0: print(f"  Epoch {ep+1}/{epochs} loss={total/len(loader):.5f}")

    out = os.path.dirname(__file__)
    torch.save(model.state_dict(), os.path.join(out,'word_spectrum_generator.pt'))
    np.save(os.path.join(out,'word_spectrum_norm.npy'), {'v_mean':v_mean,'v_std':v_std,'mel_min':mel_min,'mel_max':mel_max})
    print(f"Saved. Final loss={total/len(loader):.5f}")
    return model

if __name__ == '__main__': train()
