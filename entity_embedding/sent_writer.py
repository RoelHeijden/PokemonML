import os
from tqdm import tqdm
import ujson
import re

from evaluation_network.data.lookups import (
    POKEMON_LOOKUP,
    FORM_LOOKUP,
)


def main():
    writer = SentWriter()
    writer.write_poke_sents(file_name='sents_per_poke_1200+.txt', min_rating=1200)


class SentWriter:
    def __init__(self):
        self.data_folder = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/raw-ou-incomplete/"
        self.out_folder = "sent_files/"

    def write_poke_sents(self, file_name='sents_per_poke.txt', min_rating=0):
        files = sorted(
            [
                os.path.join(self.data_folder, file_name)
                for file_name in os.listdir(self.data_folder)
            ]
        )
        print(f'{len(files)} files collected')

        out_file = os.path.join(self.out_folder, file_name)

        with open(out_file, "w+") as out:
            for file in tqdm(files):
                with open(file, "r", encoding='utf-8') as f:
                    for line in f:
                        d = ujson.loads(line)

                        # get rating
                        _, p1_rating, p2_rating = self._get_player_ratings(d['inputLog'], file)

                        pokes = []
                        if p1_rating >= min_rating:
                            pokes += d["p1team"]
                        if p2_rating >= min_rating:
                            pokes += d["p2team"]

                        for poke in pokes:
                            sent = self._create_poke_sent(poke)
                            out.write(f"{sent}\n")

    def _get_player_ratings(self, game_input_log, battle_id):
        """" extract average rating from a game, using the inputLog """

        def is_rated_battle(input_log):
            for string in input_log:
                if string.startswith('>start'):
                    d = ujson.loads(string.strip('>start '))
                    return d.get('rated') == 'Rated battle'
            raise KeyError("key '>start' not found in input_log of battle {}".format(battle_id))

        def get_rating(player, input_log, rated_battle):
            if not rated_battle:
                return 0

            for string in input_log:
                if string.startswith('>player ' + player):
                    string = string.strip(('>player ' + player + ' '))

                    # for some reason it's stored as: {""name": ..} but idk if that is always the case
                    if string[1] == '"' and string[2] == '"':
                        string = string[:1] + string[2:]

                    d = ujson.loads(string.strip('>player ' + player + ' '))
                    return d.get("rating")
            raise KeyError("key '>player {}' not found in input_log of battle {}".format(player, battle_id))

        is_rated_battle = is_rated_battle(game_input_log)
        p1_rating = get_rating('p1', game_input_log, is_rated_battle)
        p2_rating = get_rating('p2', game_input_log, is_rated_battle)

        return is_rated_battle, p1_rating, p2_rating

    def _create_poke_sent(self, poke):
        def normalize_name(name):
            return "".join(re.findall("[a-zA-Z0-9]+", name)).replace(" ", "").lower()

        word_list = []

        name = normalize_name(poke["species"])
        if FORM_LOOKUP.get(name):
            name = FORM_LOOKUP[name]

        # species
        word_list.append(normalize_name(name))

        # item
        word_list.append(normalize_name(poke["item"]))

        # ability
        word_list.append(normalize_name(poke["ability"]))

        # moves
        for move in poke["moves"]:
            word_list.append(normalize_name(move))

        # typing
        for typing in POKEMON_LOOKUP[name]['types']:
            word_list.append(normalize_name(typing))

        return " ".join(word_list)


if __name__ == "__main__":
    main()


