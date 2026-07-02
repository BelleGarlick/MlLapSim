import os

import numpy as np
from torch.optim import NAdam
from torch.nn import HuberLoss
from torch.utils.data import DataLoader
import torch
import torch.optim.lr_scheduler as lr_scheduler
import webdataset

from lapsim.mlflow_logger.mlflow_logger import NewLogger
from lapsim.models.lapsim import LapSimModel
from lapsim.preprocessor.encoder import decode
from lapsim.evals.evaluate import evaluate
from lapsim.normalisation import TransformNormalisation
from lapsim.render import RenderItem, plot_full


BATCH_SIZE = 2048
EPOCHS = 1000
FORESIGHT = 120
SAMPLING = 4
CHECKPOINT_EVERY = 10

NORMALISATION_BOUNDS_PATH = f"bounds-v3.json"

TRAIN_DATASET = "/Users/belle/Developer/MlLapSim/dataset/processed/lapsim-train-{00..04}.tar"
VALIDATION_DATASET = "/Users/belle/Developer/MlLapSim/dataset/processed/lapsim-validation-0.tar"
TEST_DATASET = "/Users/belle/Developer/MlLapSim/dataset/processed/lapsim-test-0.tar"
# REAL_DATASET = "/Users/belle/Developer/MlLapSim/dataset/processed/lapsim-real-0.tar"

training_dataset = webdataset.WebDataset(TRAIN_DATASET).map(decode)
validation_dataset = webdataset.WebDataset(VALIDATION_DATASET).map(decode)
test_dataset = webdataset.WebDataset(TEST_DATASET).map(decode)
# real_dataset = webdataset.WebDataset(REAL_DATASET).map(decode)


bounds = TransformNormalisation()
bounds.transform.method = "flat-window"
bounds.transform.foresight = FORESIGHT
bounds.transform.sampling = SAMPLING

if os.path.exists(NORMALISATION_BOUNDS_PATH):
    bounds = TransformNormalisation.load(NORMALISATION_BOUNDS_PATH)
    print("Existing bounds loaded.")

else:
    print("No bounds file found, calculating new ones.")
    for record in training_dataset:
        bounds.extend(record)

    bounds.save(NORMALISATION_BOUNDS_PATH)
    print("Finished calculating bounds.")

# todo must set the neurons in the model and then start training

bounds.transform.single_sample = True



ls_model = LapSimModel(weights_path=None, bounds=bounds)
print("Params", ls_model.total_params)


loss = HuberLoss()
optimiser = NAdam(ls_model.model.parameters(), lr=5e-4)
scheduler = lr_scheduler.LinearLR(optimiser, start_factor=1.0, end_factor=0.005, total_iters=EPOCHS)
logger = NewLogger("lapsim", run_name="test")


def tensor(x):
    return torch.squeeze(x).to(dtype=torch.float32).to(ls_model.device)


# training_dataloader = training_dataset \
#     .shuffle(0) \
#     .map(bounds.normalise) \
#     .map(bounds.transform) \
#     .batched(BATCH_SIZE)

training_dataloader = DataLoader(
    training_dataset \
        .shuffle(0)
        .map(bounds.normalise) \
        .map(bounds.transform) \
        .batched(BATCH_SIZE),
    # batch_size=None,
    # num_workers=4,
)

validation_dataloader = DataLoader(
    validation_dataset \
        .shuffle(0)
        .map(bounds.normalise) \
        .map(bounds.transform) \
        .batched(BATCH_SIZE),
    # batch_size=None,
    # num_workers=4,
)



if __name__ == "__main__":
    best_model_perf = None

    for epoch in range(EPOCHS):
        ls_model.model.train()

        # todo move the tensorisation to the dataloader
        for i, (x, vehicles, positions, velocities) in enumerate(training_dataloader):
            optimiser.zero_grad()

            pred_pos, pred_vel = ls_model.model(
                tensor(x),
                tensor(vehicles)
            )

            pos_loss = loss(pred_pos, tensor(positions))
            vel_loss = loss(pred_vel, tensor(velocities))
            total_loss = pos_loss + vel_loss
            total_loss.backward()
            optimiser.step()

            logger.log_training_data(
                epoch,
                batch=i,
                position_loss=pos_loss.item(),
                velocity_loss=vel_loss.item(),
            )

        # Validation
        ls_model.model.eval()
        with torch.no_grad():
            for i, (x, vehicles, positions, velocities) in enumerate(validation_dataloader):
                val_pred_pos, val_pred_vel = ls_model.model(
                    tensor(x),
                    tensor(vehicles)
                )

                pos_loss = loss(val_pred_pos, tensor(positions))
                vel_loss = loss(val_pred_vel, tensor(velocities))

                logger.log_validation_data(
                    epoch,
                    batch=i,
                    position_loss=pos_loss.item(),
                    velocity_loss=vel_loss.item(),
                )

        current_val_loss = np.mean(logger.val_pos_loss)
        best_validation_reached = best_model_perf is None or current_val_loss < best_model_perf

        logger.flush(epoch, learning_rate=-1)

        if best_validation_reached:
            torch.save(ls_model.model.state_dict(), "ls3-best-val.pt")
            torch.save(optimiser.state_dict(), "ls3-best-val-optim.pt")
            best_model_perf = current_val_loss

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            bounds.transform.single_sample = False
            # Predict on the real dataset
            requests = list(test_dataset)[:1000]
            predictions = ls_model.predict(requests)
            pairs = list(zip(requests, predictions))
            evaluations = evaluate(pairs)

            for i in range(4, 5):
                plot_full(
                    tracks=[
                        RenderItem(
                            track=pairs[i][0],
                            label="Target",
                            color="green"
                        ),
                        RenderItem(
                            track=pairs[i][1],
                            label="Predicted",
                            color="red"
                        ),
                    ],
                    title="..."
                )

            # Run evals
            logger.log_checkpoint(epoch, evaluations)
            bounds.transform.single_sample = True

        scheduler.step()

