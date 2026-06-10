import mlflow
import numpy as np

from lapsim.evals.evaluation import Evaluation


class NewLogger:

    def __init__(self, experiment_name, run_name=None):
        """
        Initializes the MLflow experiment and starts a run.
        """
        self.train_pos_loss = []
        self.train_vel_loss = []
        self.val_pos_loss = []
        self.val_vel_loss = []

        self.log_every_n_batch = 10

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(experiment_name)
        # This starts a global run. In a real training script,
        # you might want to manage this with 'with mlflow.start_run():'
        # outside this class, but for a direct replacement:
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=run_name)

    def log_training_data(self,
                epoch,
                batch,
                n_batches,
                partition,
                position_loss,
                velocity_loss):
        self.train_pos_loss.append(position_loss)
        self.train_vel_loss.append(velocity_loss)

        if (batch + 1) % self.log_every_n_batch == 0:
            print(", ".join(
                [f"\rTraining. Epoch: {epoch + 1}"]
                + [f"Batch: {batch + 1}/{n_batches} {partition}"]
                + ["Position Loss: {:.6f}".format(np.mean(self.train_pos_loss))]
                + ["Velocity Loss: {:.6f}".format(np.mean(self.train_vel_loss))]
                + [" " * 10]
            ), end="")

            mlflow.set_tag("State", "Training")
            mlflow.set_tag("Epoch", epoch)
            mlflow.set_tag("Batch", f"{batch}/{n_batches}/{partition}")
            mlflow.set_tag("Avg Epoch Position Training Loss", round(np.mean(self.train_pos_loss), 5))
            mlflow.set_tag("Avg Epoch Validation Training Loss", round(np.mean(self.train_vel_loss), 5))

    def log_validation_data(self,
                epoch,
                batch,
                n_batches,
                position_loss,
                velocity_loss):
        self.val_pos_loss.append(position_loss)
        self.val_vel_loss.append(velocity_loss)

        if (batch + 1) % self.log_every_n_batch == 0:
            print(", ".join(
                [f"\rValidating. Epoch: {epoch + 1}"]
                + [f"Batch: {batch + 1}/{n_batches}"]
                + ["Position Loss: {:.6f}".format(np.mean(self.val_pos_loss))]
                + ["Velocity Loss: {:.6f}".format(np.mean(self.val_vel_loss))]
                + [" " * 10]
            ), end="")

            mlflow.set_tag("State", "Validating")
            mlflow.set_tag("Epoch", epoch)
            mlflow.set_tag("Batch", f"{batch}/{n_batches}")
            mlflow.set_tag("Avg Epoch Position Validation Loss", round(np.mean(self.val_pos_loss), 5))
            mlflow.set_tag("Avg Epoch Validation Validation Loss", round(np.mean(self.val_vel_loss), 5))

    def flush(
            self,
            epoch: int,
            # learning_rate: float,
    ):
        metrics = {
            "loss/train/position": float(np.mean(self.train_pos_loss)),
            "loss/train/velocity": float(np.mean(self.train_vel_loss)),
            "loss/validation/position": float(np.mean(self.val_pos_loss)),
            "loss/validation/velocity": float(np.mean(self.val_vel_loss)),
            # "system/learning_rate": learning_rate
        }
        mlflow.log_metrics(metrics, step=epoch)

        print(", ".join(
            [f"\rEpoch: {epoch + 1}"]
            + ["{}: {:.6f}".format(key, value) for key, value in sorted(metrics.items(), key=lambda item: item[0])]
        ))

        self.train_pos_loss.clear()
        self.train_vel_loss.clear()
        self.val_pos_loss.clear()
        self.val_vel_loss.clear()

    def log_checkpoint(self, epoch, model, sample_input, evaluation: Evaluation):
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name=f"checkpoint-epoch-{epoch}",
            step=epoch,
            input_example=sample_input,
        )

        mlflow.log_metrics(
            {
                f"eval/laptime/abs-error": evaluation.laptime.abs_error,
                f"eval/laptime/percentage": evaluation.laptime.percentage,
                f"eval/laptime/error-per-minute": evaluation.laptime.error_per_minute,
                f"eval/position/max": evaluation.position.max,
                f"eval/position/mean": evaluation.position.mean,
                f"eval/position/mean-abs": evaluation.position.mean_absolute,
                f"eval/position/rmse": evaluation.position.rmse,
                f"eval/position/ci95": evaluation.position.ci95,
                f"eval/position/percentage-mean": evaluation.position.percentage_mean,
                f"eval/position/percentage-max": evaluation.position.percentage_max,
                f"eval/position/percentage-ci95": evaluation.position.percentage_ci95,
                f"eval/position/apex-mean": evaluation.position.apex_mean,
                f"eval/position/apex-mean-abs": evaluation.position.apex_mean_absolute,
                f"eval/position/apex-max": evaluation.position.apex_max,
                f"eval/velocity/max": evaluation.velocity.max,
                f"eval/velocity/mean": evaluation.velocity.mean,
                f"eval/velocity/mean-abs": evaluation.velocity.mean_absolute,
                f"eval/velocity/rmse": evaluation.velocity.rmse,
                f"eval/velocity/ci95": evaluation.velocity.ci95,
                f"eval/velocity/percentage-mean": evaluation.velocity.percentage_mean,
                f"eval/velocity/percentage-max": evaluation.velocity.percentage_max,
                f"eval/velocity/percentage-ci95": evaluation.velocity.percentage_ci95,
                f"eval/velocity/apex-mean": evaluation.velocity.apex_mean,
                f"eval/velocity/apex-mean-abs": evaluation.velocity.apex_mean_absolute,
                f"eval/velocity/apex-max": evaluation.velocity.apex_max,
            },
            step=epoch,
            model_id=model_info.model_id
        )

        print(
            "LapTime Error:",
            round(evaluation.laptime.abs_error, 3),
            "Pos Error:",
            round(evaluation.position.mean_absolute, 3),
            "Vel Error:",
            round(evaluation.velocity.mean_absolute, 3)
        )