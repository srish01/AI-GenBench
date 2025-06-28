import csv
from pathlib import Path
import re
from typing import Callable, Dict, List, Any, Tuple


def python_function_loader(
    loader: Any, node: Any, functions: Dict[str, Callable]
) -> Callable:
    """Returns the function specified in the YAML file.
    Args:
        loader: The YAML loader instance.
        node: The YAML node representing the function name.
        functions: A dictionary mapping function names to function objects.
    Returns:
        The function corresponding to the name specified in the YAML node.
    Raises:
        ValueError: If the function name is not found in the functions dictionary.
    """
    func_name = loader.construct_scalar(node)
    try:
        return functions[func_name]
    except KeyError:
        raise ValueError(f"Function '{func_name}' not defined.")


def add_loader(yaml: Any, match: str, functions: Dict[str, Callable]) -> None:
    """Adds a custom loader to the YAML parser.
    Args:
        yaml: The YAML parser instance.
        match: The string tag that matches the function tag name in the YAML file.
        functions: A dictionary mapping function names to function objects.
    """
    yaml.constructor.add_constructor(
        match, lambda loader, node: python_function_loader(loader, node, functions)
    )


def resolve_mapping(
    resolved_value: str, mappings: List[Dict[str, Any]], context: Dict[str, Any]
) -> str:
    """Resolves the value based on mappings and context. Supports recursive mappings."""
    for mapping in mappings:
        match = mapping.get("match")
        if (isinstance(match, list) and resolved_value in match) or (
            isinstance(match, str)
            and (resolved_value == match or re.fullmatch(match, resolved_value))
        ):
            resolved_value = mapping["value"].format(**context)
            nested_mappings = mapping.get("mappings", [])
            if nested_mappings:
                resolved_value = resolve_mapping(
                    resolved_value, nested_mappings, context
                )
            break
    return resolved_value


def process_generator(
    generator_configs: Dict[str, Any],
    generator: str,
    values: List[str],
    context: Dict[str, Any] = {},
    data: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Processes the configuration of a specific generator.
    Args:
        generator_configs: The generator configurations loaded from YAML.
        generator: The name of the generator to process.
        values: List of configuration values to process.
        context: A dictionary of context values for formatting strings.
        data: A dictionary of additional data used for custom functions.
    Returns:
        A dictionary containing the processed values for each generator field.
    Raises:
        ValueError: If the generator is not found in the configuration.
    """
    config = generator_configs.get("generators", {}).get(generator)
    if not config:
        raise ValueError(f"Unknown generator: {generator}")
    final_values = {}
    for value in values:
        field_config = config.get(value, {})
        config_value = field_config.get("value")
        if callable(config_value):
            resolved_value = config_value(data)
        elif isinstance(config_value, str):
            resolved_value = config_value.format(**context)
        else:
            resolved_value = config_value
        mappings = field_config.get("mappings", [])
        resolved_value = resolve_mapping(resolved_value, mappings, context)
        final_values[value] = resolved_value
    return final_values


class DatasetContext:
    """Context class to manage values and lazy evaluations in a dataset pipeline."""

    def __init__(self):
        self._values: Dict[str, Callable] = {}

    def add_value(self, key: str, func: Callable) -> None:
        """Adds a value to the context.
        Args:
            key: The key under which the value will be stored.
            func: A callable that returns the value when evaluated.
        """
        self._values[key] = func

    def add_values(self, values: Dict[str, Callable]) -> None:
        """Adds multiple values to the context.
        Args:
            values: A dictionary of keys and corresponding callables to store in the context.
        """
        for key, func in values.items():
            self.add_value(key, func)

    def __getitem__(self, key: str) -> Any:
        """Retrieves the value associated with a key, evaluating it if necessary.
        Args:
            key: The key for which to retrieve the value.
        Returns:
            The evaluated value.
        Raises:
            KeyError: If the key is not found in the context.
        """
        if key in self._values:
            try:
                return self._values[key]()
            except Exception:
                return None
        raise KeyError(f"Key '{key}' not found in context.")

    def keys(self) -> List[str]:
        """Returns all the available keys in the context."""
        return list(self._values.keys())

    def items(self) -> List[Tuple[str, Any]]:
        """Returns all the key-value pairs in the context, with evaluated values."""
        return [(key, self[key]) for key in self.keys()]

    def __iter__(self) -> iter:
        """Makes the object iterable by keys."""
        return iter(self.keys())


def subset_size_with_sampling_per_generator(
    sample_per_generator: Dict[str, int],
    new_indices_group: Dict[str, List[int]],
) -> int:
    return sum(
        min(samples, len(new_indices_group[generator]))
        for generator, samples in sample_per_generator.items()
    )


def create_dict_from_csv(
    csv_path: Path, key_id: int = 0, value_id: int = 1
) -> Dict[str, str]:
    """
    Reads a CSV file and creates a dictionary from its contents.

    Each row in the CSV must contain at least `key_id + 1` and `value_id + 1` columns
    to be included in the resulting dictionary. The values from the column specified
    by `key_id` are used as keys, and the values from the column specified by
    `value_id` are used as dictionary values.

    Parameters:
    - csv_path (Path): The path to the CSV file.
    - key_id (int): The index of the column to use as dictionary keys (default is 0).
    - value_id (int): The index of the column to use as dictionary values (default is 1).

    Returns:
    - Dict[str, str]: A dictionary where keys and values are taken from the specified
      columns in the CSV file.

    Raises:
    - ValueError: If a row does not contain enough columns to extract the specified key or value.
    """
    data_dict = {}
    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row_index, row in enumerate(reader):
            if len(row) > max(key_id, value_id):
                key, value = str(row[key_id]), str(row[value_id])
                data_dict[key] = value
            else:
                raise ValueError(
                    f"Row {row_index} does not have enough columns. "
                    f"Expected at least {max(key_id, value_id) + 1}, got {len(row)}. "
                    f"Row content: {row}"
                )
    return data_dict


__all__ = [
    "python_function_loader",
    "add_loader",
    "process_generator",
    "DatasetContext",
    "create_dict_from_csv",
]
