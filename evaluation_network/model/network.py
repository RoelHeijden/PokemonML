import torch
from torch import nn
from typing import Dict

from evaluation_network.model.encoder import Encoder


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()

        # input sizes dependent on the StateTransformer output
        field_size = 21
        side_size = 16
        pokemon_attributes = 75 + (4 * 3)

        # encoding layer
        species_dim = 64
        item_dim = 16
        ability_dim = 16
        move_dim = 16
        self.encoding = Encoder(species_dim, item_dim, ability_dim, move_dim, load_embeddings=True)

        # pokemon layer
        pokemon_in = species_dim + item_dim + ability_dim + move_dim * 4 + pokemon_attributes
        pokemon_out = 192
        self.pokemon_layer = PokemonLayer(pokemon_in, pokemon_out, drop_rate=0.2)

        # full state layer
        state_layer_in = pokemon_out * 12 + side_size * 2 + field_size
        fc1_out = 1024
        fc2_out = 512
        state_out = 128
        self.state_layer = FullStateLayer(state_layer_in, fc1_out, fc2_out, state_out, drop_rate=0.3)

        # output later
        self.output = OutputLayer(state_out)

    def forward(self, fields: torch.tensor, sides: torch.tensor, pokemon: Dict[str, torch.Tensor]):
        # embed and concatenate pokemon variables
        fields, sides, pokemon = self.encoding(fields, sides, pokemon)

        # pokemon layer
        pokemon_out = self.pokemon_layer(pokemon)

        # state layer
        state_out = self.state_layer(
            torch.cat(
                (
                    torch.flatten(pokemon_out, start_dim=1),
                    torch.flatten(sides, start_dim=1),
                    fields
                ),
                dim=1
            )
        )

        # output layer
        output = self.output(state_out)

        return output


class PokemonLayer(nn.Module):
    def __init__(self, input_size, output_size, drop_rate):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)
        self.bn2d = nn.BatchNorm2d(2 * 6)

        self.drop = nn.Dropout(p=drop_rate)
        self.relu = nn.ReLU()

    def forward(self, x) -> torch.tensor:
        x = self.fc(x)
        x = self.relu(x)
        x = self.drop(x)

        # apply batch normalization to each individual pokemon output
        bs, d, h, w = x.shape
        x = x.view(bs, d * h, w).unsqueeze(2)
        x = self.bn2d(x).view(bs, d, h, w)

        return x


class FullStateLayer(nn.Module):
    def __init__(self, input_size, fc1_out, fc2_out, output_size, drop_rate):
        super().__init__()

        self.fc1 = nn.Linear(input_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, output_size)

        self.bn1 = nn.BatchNorm1d(fc1_out)
        self.bn2 = nn.BatchNorm1d(fc2_out)
        self.bn3 = nn.BatchNorm1d(output_size)

        self.drop = nn.Dropout(p=drop_rate)

        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.bn2(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.bn3(x)

        return x


class OutputLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.tensor):
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

