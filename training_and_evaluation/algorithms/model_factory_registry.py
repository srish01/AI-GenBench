from typing import Callable, OrderedDict


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class ModelFactoryRegistry(metaclass=SingletonMeta):
    _model_factories: OrderedDict[str, Callable] = OrderedDict()

    def register_model_factory(self, factory_name: str, factory: Callable):
        self._model_factories[factory_name] = factory

    def get_model_factories(self):
        return self._model_factories

    def make_model(
        self,
        model_name: str,
        pretrained: bool,
        **kwargs,
    ):
        for factory_name, factory in self._model_factories.items():
            try:
                result = factory(model_name=model_name, pretrained=pretrained, **kwargs)
                if result is not None:
                    return result, factory_name
            except Exception as e:
                pass
        raise ValueError(f"A factory for model {model_name} was not found")


__all__ = [
    "ModelFactoryRegistry",
]
