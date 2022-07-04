import configparser
import os


path = "../config/config.ini"


def __create_config(path: str) -> None:
    """ Создает файл конфигурации, если его еще нет"""

    if not os.path.exists(path):
        config = configparser.ConfigParser()
        config.add_section("crowler_settings")
        config.set("crowler_settings", "timeout_request", "5")
        config.set("crowler_settings", "threads_count", "5")

        # записываем параметры в файл настроек
        with open(path, "w") as config_file:
            config.write(config_file)


def get_setting(section, setting):
    """ Возвращает значение необходимого параметра из файла конфигурации"""
    __create_config(path)
    config = configparser.ConfigParser()
    config.read(path)
    value = config.get(section, setting)
    return value


if __name__ != "__main__":
    __create_config(path)
