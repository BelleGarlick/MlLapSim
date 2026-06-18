from typing import Sequence

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

    def predict(self, dataset: Sequence[dict], log_progress=True):
        """
        Predicts position and velocity of each vehicle.

        :param dataset:
        :param log_progress:
        :return:
        """
        with torch.no_grad():
            self.model.eval()

            outputs = []
            for i, record in enumerate(dataset):
                if log_progress:
                    print("\r" + record["id"] + " " * 30, end="")

                # Normalise the record and reshape it to the data the model trains upon
                normalised = self.bounds.normalise(record)
                transformed = self.bounds.transform.transform([normalised], cores=4)

                # Predict the model output
                pred_pos, pred_vel = self.model(
                    torch.tensor(transformed[0], dtype=torch.float32).to(self.device),
                    torch.tensor(transformed[1], dtype=torch.float32).to(self.device)
                )

                # Convert back to the normal format and denormalise
                # todo output as a record
                pred_pos, pred_vel = self.bounds.detransform_and_denormalise(
                    len(record["angles"]),
                    position=pred_pos.cpu().detach().numpy(),
                    velocity=pred_vel.cpu().detach().numpy()
                )

                # Construct the output format
                outputs.append({**record, "pos": pred_pos, "vel": pred_vel})

            if log_progress:
                print("\rDone" * 30)

            return outputs

