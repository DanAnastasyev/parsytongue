# -*- coding: utf-8 -*-

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class DeviceGetterMixin(object):
    @property
    def device(self):
        return next(self.parameters()).device


class BaseFactory:
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class) -> Callable:
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def build(cls, name: str, *args, **kwargs):
        selected_cls = cls.registry[name]
        return selected_cls(*args, **kwargs)
