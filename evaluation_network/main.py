import os
import torch

from training import Trainer
from testing import Tester
from model.network import ValueNet


def main():
    model = ValueNet()
    rating = 'all'
    folder = 'trained_models'
    model_name = 'EvalNet'

    train(model, rating, folder, model_name, train_new=True)
    # train(model, rating, folder, model_name, train_new=False)
    # test(model, rating, folder, model_name, test_states=True)
    # test(model, rating, folder, model_name, test_games=True)


def train(model, rating, folder, model_name, train_new=True):
    model_folder = ''
    model_file = os.path.join(model_folder, folder, model_name + '.pt')

    data_folder = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/training_states/'
    save_path = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/training_dump/'

    n_epochs = 100
    batch_size = 256
    lr = 2e-4
    lr_decay = 0.95
    lr_decay_steps = 1
    weight_decay = 0.0

    trainer = Trainer(
        model=model,
        data_folder=os.path.join(data_folder, rating),
        save_path=save_path,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        lr_decay=lr_decay,
        lr_decay_steps=lr_decay_steps,
        weight_decay=weight_decay,
        update_every_n_batches=50,
        file_size=10000,
        buffer_size=10000,
        num_workers=4,
        shuffle_data=True,
        shuffle_players=True,
        shuffle_pokemon=True,
        shuffle_moves=True,
        save_model=True
    )

    if train_new:
        trainer.train(run_name=rating, start_epoch=1)

    else:
        checkpoint = torch.load(model_file)
        trainer.model.load_state_dict(checkpoint['model'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']

        for g in trainer.optimizer.param_groups:
            g['lr'] = lr

        trainer.train(run_name=model_name + '_' + rating, start_epoch=epoch+1)


def test(model, rating, folder, model_name, test_games=False, test_states=False, test_embeddings=False):
    model_folder = ''
    model_file = os.path.join(model_folder, folder, model_name + '.pt')
    tester = Tester(model, model_file)

    if test_states:
        tester.test_states(rating)

    if test_games:
        tester.test_games(rating)


if __name__ == '__main__':
    main()

