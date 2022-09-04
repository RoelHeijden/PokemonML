import torch
import math
import json
import random
import os
import numpy as np

from data.transformer import StateTransformer
from data.data_loader import data_loader


class Tester:
    def __init__(self, model, model_file):
        self.states_folder = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/test_states/'
        self.games_folder = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/test_games/'

        self.transform = StateTransformer(shuffle_players=True, shuffle_pokemon=True, shuffle_moves=True)

        self.model = model
        self.model.load_state_dict(torch.load(model_file)['model'])

    def test_states(self, folder):
        self.model.eval()

        # init data loader
        data_path = os.path.join(self.states_folder, folder)
        dataloader = data_loader(data_path, self.transform, batch_size=256, shuffle=False)

        # track amount of evaluations and amount of correct classifications
        n_evaluations = 0
        correct_classifications = 0

        # iterate over all games
        for i, state in enumerate(dataloader, start=1):

            labels = torch.squeeze(state['result'])
            fields = state['fields']
            sides = state['sides']
            pokemon = state['pokemon']

            # forward pass
            evaluations = self.model(fields, sides, pokemon).squeeze()
            n_evaluations += len(evaluations)

            # get classification
            predictions = torch.round(evaluations)
            correct_classifications += torch.sum(labels == predictions).item()

            # print accuracy every n batches
            if i % 10 == 0:
                print(f'\r{n_evaluations} states evaluated -- average accuracy: {correct_classifications / n_evaluations:.3f}', end='')

        print(f'\r{n_evaluations} states evaluated')
        print(f'average accuracy: {correct_classifications / n_evaluations:.3f}\n')

    def test_games(self, folder):
        self.model.eval()

        # init arrays tracking the data
        self.step_size = 5

        # track n games
        n_games = 0
        self.n_games_per_length_range = np.zeros(6)
        self.n_games_per_rating_range = np.zeros(4)

        # track game rating and length
        self.game_ratings = []
        self.game_lengths = []

        # average evaluation
        self.win_evaluations = np.zeros(int(100 / self.step_size))
        self.loss_evaluations = np.zeros(int(100 / self.step_size))
        self.n_wins = np.zeros(int(100 / self.step_size))
        self.n_losses = np.zeros(int(100 / self.step_size))

        # overall accuracy
        self.n_correct = np.zeros(int(100 / self.step_size))
        self.n_evals = np.zeros(int(100 / self.step_size))

        # accuracy per length range
        self.n_correct_per_length_range = [np.zeros(int(100 / self.step_size)) for i in range(6)]
        self.n_evals_per_length_range = [np.zeros(int(100 / self.step_size)) for i in range(6)]

        # accuracy per rating range
        self.n_correct_per_rating_range = [np.zeros(int(100 / self.step_size)) for i in range(4)]
        self.n_evals_per_rating_range = [np.zeros(int(100 / self.step_size)) for i in range(4)]

        # init data files
        data_path = os.path.join(self.games_folder, folder)
        files = sorted([os.path.join(data_path, file_name)
                        for file_name in os.listdir(data_path)])

        random.shuffle(files)
        min_game_length = 3

        # iterate over and open each game in the test games folder
        for n_games, f in enumerate(files, start=1):
            with open(f) as f_in:
                game_states = json.load(f_in)

                # skip game if too short
                if len(game_states) < min_game_length:
                    continue

                results = []
                evaluations = []
                percentage_completed = []

                # iterate all game states, starting at 1 to avoid team preview states
                for i in range(1, len(game_states)):
                    state = game_states[i]

                    # run state through evaluation network
                    evaluation, game_result = self._evaluate_state(state)

                    # store evaluation
                    evaluations.append(round(evaluation, 3))

                    # store game result each state because the player pov can be shuffled in transformer
                    results.append(game_result)

                    # store game % completed
                    percentage_completed.append((i - 1) / (len(game_states) - 1) * 100)

                # store data into arrays of size (100 / step_size)
                self._store_results(game_states, percentage_completed, results, evaluations)

                # plot data
                if n_games % 1000 == 0:
                    self._plot_performance(n_games)

        self._plot_performance(n_games)

    def _store_results(self, game_states, percentage_completed, results, evaluations):
        game_length = game_states[-1]['turn']
        length_index = min(math.floor(game_length / 10), 5)

        rating = min(game_states[0]['p1rating'], game_states[0]['p2rating'])
        rating_index = min(math.floor((rating - 1000) / 200), 3)

        self.n_games_per_rating_range[rating_index] += 1
        self.n_games_per_length_range[length_index] += 1

        # iterate over each array slot
        for i in range(2, 100, self.step_size):
            i += 0.5

            array_idx = int((i - 2) / self.step_size)

            # map the game%completed to 20 indices, representing a percentage range {2.5, 5, 7.5, ..., 97.5}
            nearest_percentage_index = min(
                range(len(percentage_completed)),
                key=lambda j: abs(percentage_completed[j] - i)
            )

            # compare evaluation with result
            result = results[nearest_percentage_index]
            evaluation = evaluations[nearest_percentage_index]
            correct_pred = int(round(evaluation) == result)

            # store evaluations
            if result == 1:
                self.win_evaluations[array_idx] += evaluation
                self.n_wins[array_idx] += 1
            else:
                self.loss_evaluations[array_idx] += evaluation
                self.n_losses[array_idx] += 1

            # store predictions
            self.n_correct[array_idx] += correct_pred
            self.n_evals[array_idx] += 1

            # store predictions per length range
            self.n_correct_per_length_range[length_index][array_idx] += correct_pred
            self.n_evals_per_length_range[length_index][array_idx] += 1

            # store predictions per rating range
            self.n_correct_per_rating_range[rating_index][array_idx] += correct_pred
            self.n_evals_per_rating_range[rating_index][array_idx] += 1

    def _plot_performance(self, n_games):
        print(f'\n------------------- {n_games} games evaluated -------------------\n')

        # plot evaluations per label
        win_evaluations = np.round(self.win_evaluations / self.n_wins, decimals=3)
        loss_evaluations = np.round(self.loss_evaluations / self.n_losses, decimals=3)
        print(f'average evaluations: \n'
              f'wins: {", ".join(str(x) for x in win_evaluations)}\n'
              f'losses: {", ".join(str(x) for x in loss_evaluations)}\n')

        # overall accuracy
        accuracies = np.round(self.n_correct / self.n_evals, decimals=3)
        print(f'overall accuracy: {", ".join(str(x) for x in accuracies)}\n')

        # plot accuracies per length range {2-9, 10-19, 20-29, 30-39, 40-49, 50+}
        for i in range(6):
            accuracies_per_length = np.round(self.n_correct_per_length_range[i] / self.n_evals_per_length_range[i], 3)
            print(f'accuracies length range {i}: {", ".join(str(x) for x in accuracies_per_length)}')
        print(f'n games per length range: {self.n_games_per_length_range}\n')

        # plot accuracies per rating
        for i in range(4):
            accuracies_per_rating = np.round(self.n_correct_per_rating_range[i] / self.n_evals_per_rating_range[i], 3)
            print(f'accuracies rating range {i}: {", ".join(str(x) for x in accuracies_per_rating)}')
        print(f'n games per rating range: {self.n_games_per_rating_range}\n')

    def _evaluate_state(self, state):
        # transform state into dict of tensors
        tensor_dict = self.transform(state)

        # get label
        game_result = int(tensor_dict['result'].item())

        # network is hardcoded for batches, so create batch of size 1
        fields = torch.unsqueeze(tensor_dict['fields'], 0)
        sides = torch.unsqueeze(tensor_dict['sides'], 0)
        pokemon = {key: torch.unsqueeze(value, 0) for key, value in tensor_dict['pokemon'].items()}

        # forward pass
        evaluation = self.model(fields, sides, pokemon).item()

        return evaluation, game_result



