import torch
from torch import nn
from typing import Dict
import fasttext
import os
import numpy as np

from evaluation_network.data.categories import (
    SPECIES,
    MOVES,
    ITEMS,
    ABILITIES
)


class Encoder(nn.Module):
    def __init__(self, species_dim, item_dim, ability_dim, move_dim, load_embeddings=False):
        super().__init__()

        if not load_embeddings:

            self.species_embedding = nn.Embedding(len(SPECIES) + 1, species_dim, padding_idx=0)
            self.item_embedding = nn.Embedding(len(ITEMS) + 1, item_dim, padding_idx=0)
            self.ability_embedding = nn.Embedding(len(ABILITIES) + 1, ability_dim, padding_idx=0)
            self.move_embedding = nn.Embedding(len(MOVES) + 1, move_dim, padding_idx=0)

        else:

            species_weights = self._init_custom_embedding(species_dim, SPECIES)
            item_weights = self._init_custom_embedding(item_dim, ITEMS)
            ability_weights = self._init_custom_embedding(ability_dim, ABILITIES)
            move_weights = self._init_custom_embedding(move_dim, MOVES)

            self.species_embedding = nn.Embedding.from_pretrained(species_weights, freeze=True, padding_idx=0)
            self.item_embedding = nn.Embedding.from_pretrained(item_weights, freeze=True, padding_idx=0)
            self.ability_embedding = nn.Embedding.from_pretrained(ability_weights, freeze=True, padding_idx=0)
            self.move_embedding = nn.Embedding.from_pretrained(move_weights, freeze=True, padding_idx=0)

    def _init_custom_embedding(self, n_dims, category):
        model_folder = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/Python/PokemonML/entity_embedding/model_files/'

        # load embedding model
        model_file = os.path.join(model_folder, 'poke_embeddings_' + str(n_dims) + '_dim.bin')
        model = fasttext.load_model(model_file)

        # concat embedding vector of each entity
        vectors = [model.get_word_vector(x) for x in category.keys()]

        # insert empty array at index 0 -- category lists start at index=1
        vectors.insert(0, np.zeros(n_dims, dtype=float))

        weights = torch.FloatTensor(vectors)
        return weights

    def _concat_pokemon(self, pokemon: Dict[str, torch.tensor]) -> torch.tensor:
        """
        returns: tensor of size [batch, 2, 6, pokemon]
        """
        species = self.species_embedding(pokemon["species"])
        moves = self.move_embedding(pokemon["moves"])
        items = self.item_embedding(pokemon["items"])
        abilities = self.ability_embedding(pokemon["abilities"])

        return torch.cat(
                (
                    species,
                    items,
                    abilities,
                    torch.flatten(moves, start_dim=3),
                    torch.flatten(pokemon['move_attributes'], start_dim=3),
                    pokemon['pokemon_attributes'],
                ),
                dim=3
            )

    def forward(self, fields, sides, pokemon):
        return fields, sides, self._concat_pokemon(pokemon)


