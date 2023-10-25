import locale
import json
import logging
import os

from pathlib import Path


def load_language_list(language, language_file_path):
    with open(language_file_path, "r", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list


class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[
                0
            ]  # getlocale can't identify the system's language ((None, None))
        base_dir = Path(__file__).parent.resolve()
        language_file_path = os.path.join(base_dir, f"lib/i18n/{language}.json")
        if not os.path.exists(language_file_path):
            logging.info(f"No match was found for {language}, fallback to en_US")
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language, language_file_path)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def print(self):
        print("Use Language:", self.language)