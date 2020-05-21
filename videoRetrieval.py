import torch
from model import CNN
import pickle
from utils import get_longest_shot, pic_norm

def get_full_path(base_path, entry):
    return base_path + "/" + entry[1] + "/" + entry[2]


def cosine_similarity(input_embeddings, target_embeddings):
    scores = []
    for input_embedding, target_embedding in zip(input_embeddings, target_embeddings):
        scores.append(torch.cosine_similarity(input_embedding, target_embedding, 0).item())
    return sum(scores) / len(scores)

def calc_score(model, input_video, target_video):
    model.eval()

    input_embeddings = []
    for input_path in input_video:
        img = pic_norm(input_path, (96,72))
        with torch.no_grad():
            x = torch.unsqueeze(torch.Tensor(img), 0).to("cuda")
            _, embedding = model(x)
            input_embeddings.append(embedding.detach().squeeze())

    target_embeddings = []
    for target_path in target_video:
        img = pic_norm(target_path, (96,72))
        with torch.no_grad():
            x = torch.unsqueeze(torch.Tensor(img), 0).to("cuda")
            _, embedding = model(x)
            target_embeddings.append(embedding.detach().squeeze())

    return cosine_similarity(input_embeddings, target_embeddings)

if __name__ == "__main__":
    model = CNN()
    model.load_state_dict(torch.load("./data/model.pt"))
    model.to("cuda")

    with open("./data/HMDB_search_split.pkl", "rb") as file:
        search_split = pickle.load(file)
    with open("./data/HMDB_corpus_split.pkl", "rb") as file:
        corpus_split = pickle.load(file)

    base_path = "./data/HMDB_main_frame"
    input_video = get_longest_shot(get_full_path(base_path, search_split[0]))

    scores = []
    for entry in corpus_split:
        target_video = get_longest_shot(get_full_path(base_path, entry))
        scores.append((
            calc_score(model, input_video, target_video),
            entry
        ))

    print(scores[0][0], type(scores[0][0]))
    scores = sorted(scores, key=lambda x:x[0], reverse=True)[:5]

    for score in scores:
        print("score : {}   type : {}   name : {}".format(score[0], score[1][1], score[1][2]))


