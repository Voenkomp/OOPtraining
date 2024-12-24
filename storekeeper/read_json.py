import json


def return_dict():
    with open(
        "database_connection_settings.json", encoding="UTF-8", mode="r"
    ) as readable:
        text = readable.read()
        settings = json.loads(text)
    return settings
