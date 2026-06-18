import torch
from torch import nn
import webdataset

from lapsim.models.dense_nn import LapSimModelDense


def hard_sigmoid(x):
    return torch.clamp((x + 2.5) / 5, min=0, max=1)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class LapSimModel(nn.Module):

    def __init__(self, weights_path, bounds):
        super().__init__()

        self.device = get_device()
        self.model = LapSimModelDense().to(self.device)

        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

        self.bounds = bounds

    @property
    def total_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def predict(self, dataset: webdataset.WebDataset):
        with torch.no_grad():
            self.model.eval()

            x, (y_pos, y_vel), vehicles = self.bounds.normalise_and_transform(dataset)

            pred_pos, pred_vel = self.model(
                torch.tensor(x, dtype=torch.float32).to(self.device),
                torch.tensor(vehicles, dtype=torch.float32).to(self.device)
            )

            pred_pos, pred_vel = self.bounds.detransform_and_denormalise(
                len(dataset.angles[0]),
                position=pred_pos.cpu().detach().numpy(),
                velocity=pred_vel.cpu().detach().numpy()
            )

            return pred_pos, pred_vel

