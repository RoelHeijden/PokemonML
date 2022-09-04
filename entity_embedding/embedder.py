import os
import fasttext


def main():
    embedder = Embedder()

    # embedder.create_model(poke_file='sents_per_poke_1200+.txt', dims=4)
    # embedder.create_model(poke_file='sents_per_poke_1200+.txt', dims=8)
    # embedder.create_model(poke_file='sents_per_poke_1200+.txt', dims=12)
    # embedder.create_model(poke_file='sents_per_poke_1200+.txt', dims=16)
    # embedder.create_model(poke_file='sents_per_poke_1200+.txt', dims=24)
    # embedder.create_model(poke_file='sents_per_poke_1200+.txt', dims=32)
    # embedder.create_model(poke_file='sents_per_poke_1200+.txt', dims=48)
    # embedder.create_model(poke_file='sents_per_poke_1200+.txt', dims=64)
    # embedder.create_model(poke_file='sents_per_poke_1200+.txt', dims=96)
    # embedder.create_model(poke_file='sents_per_poke_1200+.txt', dims=128)

    # embedder.test_embedding_dims('blastoise', SPECIES)
    # embedder.test_embedding_dims('figyberry', ITEMS)
    # embedder.test_embedding_dims('hex', MOVES)
    # embedder.test_embedding_dims('drought', ABILITIES)


class Embedder:
    def __init__(self):
        self.model_folder = "model_files/"
        self.sents_folder = "sent_files/"

    def create_model(self, poke_file='sents_per_poke.txt', dims=32):
        sents_file = os.path.join(self.sents_folder, poke_file)
        model = fasttext.train_unsupervised(sents_file, model="cbow", dim=dims, ws=50, minn=0, maxn=0)

        out_file = os.path.join(self.model_folder, 'poke_embeddings_' + str(dims) + '_dim.bin')
        model.save_model(out_file)

        print(f'model saved at:\n'
              f'{out_file}')

    def test_embedding_dims(self, entity, category, dims=(4, 8, 12, 16, 24, 32, 48, 64, 96, 128)):
        for dim in dims:
            file_name = 'poke_embeddings_' + str(dim) + '_dim.bin'

            model_file = os.path.join(self.model_folder, file_name)
            model = fasttext.load_model(model_file)

            print('dims:', dim)
            self.show_most_similar(entity, model, category.keys())

    def show_most_similar(self, term, model, vocabulary, num_terms=10):
        vocabulary = set(vocabulary)
        neighbors = model.get_nearest_neighbors(term, k=100)
        keep = [(sim, word) for (sim, word) in neighbors if word in vocabulary]

        print(term)
        for result in keep[:num_terms]:
            print(result)
        print()


if __name__ == "__main__":
    main()

