import torch as t
import torch.nn as nn
import torchvision as tv

def weight_name(k):
    if "encoder." in k:
        k = k.split("encoder.")[-1]
    if "net." in k:
        k = k.split("net.")[-1]
    return k

class Encoder(nn.Module):

    def __init__(self, path=None):
        super().__init__()
        self.net = tv.models.resnet18()
        self.net.fc = nn.Identity()

        # grayscale input
        conv = self.net.conv1
        assert conv.bias == None
        OutChannels, InChannels, *KernelSize = conv.weight.shape
        self.net.conv1 = nn.Conv2d(1, OutChannels, KernelSize,
                                   stride=conv.stride,
                                   padding=conv.padding,
                                   bias=False)

        if path is not None:
            ckpt = t.load(path)
            state_dict = {weight_name(k): v
                          for k, v in ckpt['state_dict'].items()
                          if "encoder." in k}
            self.net.load_state_dict(state_dict)
            print(f"Loaded state from {path}")

    def _forward(self, X):
        assert X.ndim == 4
        H = self.net(X)
        return H

    def forward(self, X):
        if X.ndim == 4:
            return self._forward(X)
        else:
            B, N, C, H, W = X.shape
            X = self._forward(X.reshape(B * N, C, H, W))
            return X.reshape((B, N, -1))


class Classifier(nn.Module):

    def __init__(self, D, K):
        super().__init__()
        self.net = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(D, D // 2),
                nn.BatchNorm1d(D // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(D // 2, K))

    def _forward(self, X):
        assert X.ndim == 2
        return self.net(X)

    def forward(self, X):
        if X.ndim == 2:
            P = self._forward(X)
        else:
            B, N, D = X.shape
            X = self._forward(X.reshape(B * N, D))
            P = X.reshape((B, N, -1))
        return P.squeeze(-1)


class GroupModel(nn.Module):

    def __init__(self, K,
                 lr=1e-4,
                 wd=1e-4,
                 encoder_path=None,
                 freeze_encoder=False):
        super().__init__()
        if freeze_encoder:
            assert encoder_path is not None

        self.encoder = Encoder(encoder_path)
        self.classifier = Classifier(D=512, K=K)
        self.loss = nn.CrossEntropyLoss()
        self.freeze_encoder = freeze_encoder

        # We cannot replace this by a call to `freeze()`, since that would
        # switch the encoder to eval mode, which would mess up the batch norm
        # statistics computation
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def encode(self, X):
        if not self.freeze_encoder:
            return self.encoder(X)
        with t.no_grad():
            return self.encoder(X)

    # Returns group predictions in the form of logits, use softmax to turn them
    # into probabilities
    def forward(self, X):
        BatchSize, GroupSize, Channels, Height, Width = X.shape
        # Instance-level prediction
        T = self.classifier(self.encode(X))
        # Group-level prediction
        S = t.sum(T, dim=1)
        return S

