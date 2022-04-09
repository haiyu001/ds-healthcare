from typing import Dict, Any, List, Tuple, Optional
from configparser import ConfigParser, ExtendedInterpolation


def _get_config_parser() -> ConfigParser:
    config = ConfigParser(interpolation=ExtendedInterpolation(), allow_no_value=True)
    config.optionxform = str
    return config


def read_config(config_filepath: str) -> ConfigParser:
    config = _get_config_parser()
    with open(config_filepath) as input:
        conf_content = input.read()
    config.read_string(conf_content)
    return config


def copy_config(config_filepath: str, save_filepath: str):
    with open(config_filepath) as input:
        conf_content = input.read()
    with open(save_filepath, "w") as output:
        output.write(conf_content)


def is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    else:
        return True


def str_to_bool(text: str) -> Optional[bool]:
    text = text.lower()
    if text in {"true", "yes", "y", "on"}:
        return True
    elif text in {"false", "no", "n", "off"}:
        return False
    else:
        return None


def clean_config_str(text: str) -> str:
    if text:
        text = text.strip(" '\"")
        text = ",".join([i.strip(" '\"") for i in text.split(",")])
    return text


def config_type_casting(config_items: List[Tuple[str, str]]) -> Dict[str, Any]:
    config_dict = {}
    for key, value_str in config_items:
        value_str = clean_config_str(value_str)
        if value_str == "":
            value = None
        elif str_to_bool(value_str) is not None:
            value = str_to_bool(value_str)
        elif is_float(value_str):
            value = float(value_str)
        elif value_str.isdigit():
            value = int(value_str)
        else:
            value = value_str
        config_dict[key] = value
    return config_dict
