import torchvision


class Models():
    def __init__(self, type, name):
        self.model_name = name
        self.model_type = type

    def load_model(self):
        return getattr(getattr(torchvision.models, self.model_type), self.model_name)
