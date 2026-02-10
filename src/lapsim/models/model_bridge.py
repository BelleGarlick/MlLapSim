from pathlib import Path

from lapsim.normalisation.transform_normalisation import TransformNormalisation


class ModelBridge:

    def __init__(self, model_path):
        super().__init__()
        self._model_root_path = Path(model_path).parent
        self.model_path = self._model_root_path / 'model.onnx'

    def get_transform(self):
        return TransformNormalisation.load(self._model_root_path / 'bounds.json')
