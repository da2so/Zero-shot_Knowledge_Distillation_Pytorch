import torch
import torch.nn as nn

from torch.nn.functional import relu


class ResLinear(nn.Module):
    def __init__(self, in_planes: int, planes: int, expansion: int = 1) -> None:

        super().__init__()
        self.fc1 = nn.Linear(in_planes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.out_planes = out_planes = expansion * planes
        self.fc2 = nn.Linear(planes, out_planes)
        self.bn2 = nn.BatchNorm1d(out_planes)

        self.shortcut = nn.Sequential(
            nn.Linear(in_planes, out_planes, bias=False),
            nn.BatchNorm1d(out_planes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(ResMLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        hidden_dim = 128
        expansion = 2
        self.input_projection = nn.Linear(self.input_size, hidden_dim)
        self.nets = nn.ModuleList()
        self.nets.append(ResLinear(hidden_dim, hidden_dim, expansion=expansion))
        self.nets.append(
            ResLinear(self.nets[-1].out_planes, hidden_dim, expansion=expansion) # type: ignore
        )
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.nets[-1].out_planes, self.output_size) # type: ignore
        self.reset_parameters()

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def reset_parameters(self) -> None:
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, returnt="out") -> torch.Tensor:
        x = x.flatten(1, -1)
        feats = self.input_projection(x)
        for net in self.nets:
            feats = net(feats)
            feats = self.dropout(feats)

        if returnt == "features":
            return feats

        out = self.classifier(feats)

        if returnt == "out":
            return out
        elif returnt == "all":
            return (out, feats)  # type: ignore

        raise NotImplementedError("Unknown return type")

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, returnt="features")

    def get_params(self) -> torch.Tensor:
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[
                progress : progress + torch.tensor(pp.size()).prod()
            ].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        return torch.cat(self.get_grads_list())

    def get_grads_list(self):
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))  # type: ignore
        return grads
