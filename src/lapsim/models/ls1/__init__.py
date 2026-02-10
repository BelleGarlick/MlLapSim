import numpy as np
import onnxruntime as ort

from lapsim.models import ModelBridge


class LS1(ModelBridge):

    def __init__(self):
        super().__init__(__file__)

        self.ort_sess = ort.InferenceSession(self.model_path)

    def predict(self, x):
        posses, vels = np.zeros((2, len(x), 9))

        for batch_idx in range(len(x)):
            pos, vel = self.ort_sess.run(
                None,
                {'onnx::Gemm_0': x[batch_idx:batch_idx + 1]}
            )

            posses[batch_idx] = pos
            vels[batch_idx] = vel

        return posses, vels
