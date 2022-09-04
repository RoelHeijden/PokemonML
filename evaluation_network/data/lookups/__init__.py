import json
import os


def init_lookup(path, file_name):
    file = os.path.join(path, file_name)
    with open(file, 'r') as f_in:
        data = json.load(f_in)
    return data


PATH = os.path.dirname(os.path.abspath(__file__))

POKEMON_LOOKUP = init_lookup(PATH, 'pokemon_lookup.json')
MOVE_LOOKUP = init_lookup(PATH, 'move_lookup.json')
FORM_LOOKUP = init_lookup(PATH, 'form_lookup.json')

VOLATILES_TO_IGNORE = init_lookup(PATH, 'volatiles_to_ignore.json')
INVULNERABLE_STAGES = init_lookup(PATH, 'invulnerable_stages.json')
VULNERABLE_STAGES = init_lookup(PATH, 'vulnerable_stages.json')

