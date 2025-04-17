from argparse import ArgumentParser
import yaml
import os


# ConfigDict is a subclass of dict that allows you to access keys as attributes
# TODO JL 1/7/25: handle constants like pi
class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert all dict values into ConfigDict instances
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key):
        return self.get(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def copy(self):
        copied_dict = ConfigDict()
        for key, value in self.items():
            if isinstance(value, dict):
                copied_dict[key] = value.copy()
            else:
                copied_dict[key] = value
        return copied_dict

    def override(self, other):
        for key, value in other.items():
            if key in self and isinstance(self[key], dict) and isinstance(value, dict):
                # Merge the dictionaries: keys in self but not in value are preserved.
                merged = dict(self[key])
                merged.update(value)  # update with the new keys
                self[key] = ConfigDict(merged)
            else:
                self[key] = value

    def update_from_dot_notation(self, key_path, value):
        """Update config using dot notation (e.g., 'model.hidden_size')"""
        keys = key_path.split(".")
        current = self
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = ConfigDict()
            current = current[key]
        current[keys[-1]] = value


def load_config(path):
    with open(path, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    return ConfigDict(config)


def parse_override(overrides):
    """Parse a list of key=value strings into a dictionary"""
    parsed_overrides = {}
    for s in overrides.split(" "):
        if "=" not in s:
            raise ValueError(f"Override argument must be in format key=value, got {s}")
        key, value = s.split("=", 1)

        # Try to infer the type of value
        try:
            # Try as int
            value = int(value)
        except ValueError:
            try:
                # Try as float
                value = float(value)
            except ValueError:
                # Try as bool
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                # Else if string is "none"
                elif value.lower() == "none":
                    value = None
                # Otherwise keep as string
        parsed_overrides[key] = value
    return parsed_overrides


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_config", type=str, required=False)
    parser.add_argument("--task_config", type=str, required=False)
    parser.add_argument("--save_path", type=str, required=False)
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help='Override config values, e.g., "model.hidden_size=256 training.lr=0.001"',
    )

    CONFIG_DIR = "/pscratch/sd/j/jwl50/nonlinear_rnns_dev/src/configs"
    args = parser.parse_args()

    # Load config
    config = load_config(os.path.join(CONFIG_DIR, "base.yaml"))

    # Override config with model and task
    if args.model_config:
        model_config = load_config(os.path.join(CONFIG_DIR, args.model_config))
        config.override(model_config)
    if args.task_config:
        task_config = load_config(os.path.join(CONFIG_DIR, args.task_config))
        config.override(task_config)
    if args.save_path:
        config.logger.path = args.save_path

    # Apply command line overrides
    for override in args.override:
        key, value = parse_override(override)
        config.update_from_dot_notation(key, value)

    print(config)
