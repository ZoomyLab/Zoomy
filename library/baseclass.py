import yaml
import os

main_dir = os.getenv('SMS')


class BaseYaml(yaml.YAMLObject):
    yaml_tag = "!Baseclass"

    def __init__(self, **kwargs):
        self.set_default_parameters()
        self.set_optional_arguments(kwargs)
        self.save_above_parameters_for_config()
        # runtime variables will only be set via explicit calling or reinitialization via read_class_from_file
        # self.set_runtime_variables()

    def set_default_parameters(self):
        return

    def set_runtime_variables(self):
        return

    def set_optional_arguments(self, kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print("attribute: ", key, " not found.")

    def save_above_parameters_for_config(self):
        self.yaml_keys = list(self.__dict__.keys())

    @classmethod
    def from_yaml(cls, loader, node):
        node_map = loader.construct_mapping(node, deep=True)
        # for node in node_map.values():
        #     if issubclass(node.__class__, BaseYaml):
        #         node.set_runtime_variables()
        value = cls(**node_map)
        value.set_runtime_variables()
        yield value

    def write_class_to_file(
        self, filepath=(main_dir + "/output/"), filename="config.yaml"
    ):
        os.makedirs(filepath, exist_ok=True)
        stream = open(os.path.join(filepath, filename), "w+")
        yaml.dump(self, stream, default_flow_style=False, sort_keys=False)
        stream.close()

    def read_class_from_file(filepath=main_dir + "/output/", filename="config.yaml"):
        read_class = yaml.load(
            open(os.path.join(filepath, filename), "rb"), Loader=BaseYaml.get_loader()
        )
        return read_class

    def get_loader():
        loader = yaml.FullLoader
        # loader = yaml.Loader
        # loader = yaml.SafeLoader
        # loader.add_constructor(BaseYaml.yaml_tag, BaseYaml.from_yaml)
        # loader.add_constructor(BaseYaml.yaml_tag, BaseYaml.from_yaml)
        return loader

    def __getstate__(self):
        state = self.__dict__.copy()
        output = {k: state[k] for k in self.yaml_keys}
        return output


class ExampleClass(BaseYaml):
    yaml_tag = "!ExampleClass"

    def set_default_parameters(self):
        self.a = 1
        self.b = 2
        self.d = 10
        self.e = [1, 2, 3]

    def set_runtime_variables(self):
        self.runtime = {}


class ExampleClass2(BaseYaml):
    yaml_tag = "!ExampleClass2"

    def set_default_parameters(self):
        self.a = 10
        self.d = "asdf"
        self.e = [1, 2, 3, 3]

    def set_runtime_variables(self):
        self.runtime = [1, 2, 3]


class ExampleClassStacked(BaseYaml):
    yaml_tag = "!ExampleClassStacked"

    def set_default_parameters(self):
        self.c1 = ExampleClass(a=2, b=-2)
        self.c2 = ExampleClass2(a=100, b=1000)


