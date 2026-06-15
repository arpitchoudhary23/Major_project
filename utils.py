import json

def save_classes(class_names):

    with open("classes.json", "w") as f:
        json.dump(class_names, f)

def load_classes():

    with open("classes.json", "r") as f:
        return json.load(f)