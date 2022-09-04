import json
import os


def init_category(path, file_name):
    """ opens a category file and converts it to a dictionary with the position index as value """
    file = os.path.join(path, file_name)
    with open(file, 'r') as f_in:
        data = json.load(f_in)
    return data


PATH = os.path.dirname(os.path.abspath(__file__))

SPECIES = init_category(PATH, 'species.json')
ITEMS = init_category(PATH, 'items.json')
ABILITIES = init_category(PATH, 'abilities.json')
MOVES = init_category(PATH, 'moves.json')
WEATHERS = init_category(PATH, 'weathers.json')
TERRAINS = init_category(PATH, 'terrains.json')
TYPES = init_category(PATH, 'types.json')
STATUS = init_category(PATH, 'status.json')
MOVE_CATEGORIES = init_category(PATH, 'move_categories.json')
VOLATILE_STATUS = init_category(PATH, 'volatile_statuses.json')
SIDE_CONDITIONS = init_category(PATH, 'side_conditions.json')


