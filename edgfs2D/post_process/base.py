# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from argparse import Namespace


class BasePostProcessor(object, metaclass=ABCMeta):
    kind = None

    def __init__(self, plugin_args: Namespace, *args, **kwargs):
        self._args = plugin_args

    @abstractmethod
    def execute(self):
        pass

    @property
    def args(self):
        return self._args

    @abstractmethod
    def parse_args(self):
        pass
