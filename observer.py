# -*- coding: utf-8 -*-
# https://www.protechtraining.com/blog/post/tutorial-the-observer-pattern-in-python-879


class Subscriber:
    def __init__(self, name):
        self.name = name
        print("Subscriber name {}".format(self.name))

    def update(self, event, message):
        print("no override")
        print('{} got message "{}"'.format(self.name, message))


class Publisher:
    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self.events = {event: dict()
                       for event in events}
        print("Publisher events {}".format(self.events))

    def get_subscribers(self, event):
        return self.events[event]

    def register(self, event, who, callback=None):
        if callback is None:
            callback = getattr(who, 'update')
        self.get_subscribers(event)[who] = callback

    def unregister(self, event, who):
        del self.get_subscribers(event)[who]

    def dispatch(self, event, message):
        for subscriber, callback in self.get_subscribers(event).items():
            callback(event, message)
