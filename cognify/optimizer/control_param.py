import dataclasses
import os
import json
import importlib

from cognify.optimizer.core.flow import LayerConfig

@dataclasses.dataclass
class ControlParameter:
    opt_layer_configs: list[LayerConfig]
    opt_history_log_dir: str = "opt_results"
    quality_constraint: float = 1.0
    train_down_sample: int = 0
    val_down_sample: int = 0
    evaluator_batch_size: int = 10

    @classmethod
    def from_python_profile(cls, param_path):
        if not os.path.isfile(param_path):
            raise FileNotFoundError(
                f"The control param file {param_path} does not exist."
            )
        module_name = os.path.basename(param_path).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, param_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, cls):
                return obj
        raise ValueError(f"No ControlParameter instance found in {param_path}")

    @classmethod
    def from_json_profile(cls, param_path):
        if not os.path.isfile(param_path):
            raise FileNotFoundError(
                f"The control param file {param_path} does not exist."
            )
        with open(param_path, "r") as f:
            param_dict = json.load(f)
        attrs = [
            attr.name
            for attr in dataclasses.fields(cls)
            if attr.name not in ["opt_layer_configs"]
        ]
        opt_layer_configs = [
            LayerConfig.from_dict(layer_dict)
            for layer_dict in param_dict["opt_layer_configs"]
        ]
        return cls(
            opt_layer_configs=opt_layer_configs,
            **{attr: param_dict[attr] for attr in attrs},
        )

    @classmethod
    def build_control_param(cls, param_path=None, loaded_module=None):
        assert (
            param_path or loaded_module
        ), "Either param_path or loaded_module should be provided."

        # prioritize param_path if given
        if param_path is not None:
            if param_path.endswith(".py"):
                control_param = ControlParameter.from_python_profile(param_path)
            else:
                control_param = ControlParameter.from_json_profile(param_path)
            return control_param
        else:
            for name in dir(loaded_module):
                obj = getattr(loaded_module, name)
                if isinstance(obj, cls):
                    return obj
            raise ValueError(f"No ControlParameter instance found in {loaded_module}")

    def to_dict(self):
        result = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if hasattr(value, "to_dict"):  # If the field has a `to_dict` method
                result[field.name] = value.to_dict()
            elif isinstance(value, list):  # Handle lists of complex objects
                result[field.name] = [
                    item.to_dict() if hasattr(item, "to_dict") else item
                    for item in value
                ]
            else:  # Handle primitive types or unsupported objects
                result[field.name] = value
        return result
