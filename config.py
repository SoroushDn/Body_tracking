import json

class BodyTrackingConfig:
    def __init__(self, dataset_image_dir="", points_dir="", num_features=""):
        self.dataset_image_dir = dataset_image_dir
        self.points_dir = points_dir
        self.num_features = num_features

    @classmethod
    def from_dict(cls, json_object):
        config = BodyTrackingConfig()
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))