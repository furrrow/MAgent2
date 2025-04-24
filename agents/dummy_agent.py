import numpy as np

class DummyAgent():
    def __init__(self, name, dummy_action = 0, dummy_message = 0):
        self.name = name
        self.dummy_action = dummy_action
        self.dummy_message = dummy_message

    def get_action(self, observation):
        return self.dummy_action, self.dummy_message
