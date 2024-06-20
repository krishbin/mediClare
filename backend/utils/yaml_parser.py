import yaml

class YMLParser:
    def __init__(self, path):
        self.path = path
        self.data = self.load_yaml()

    def load_yaml(self):
        with open(self.path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    def set(self, key, value):
        self.data[key] = value
        self.save_yaml(self.path, self.data)

    def get(self, key):
        return self.data[key]
    
    def available_keys(self):
        return self.data.keys()

    def get_all(self):
        return self.data
    
    def save_yaml(self, path, data):
        with open(path, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)