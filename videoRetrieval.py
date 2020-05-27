import torch
from model import CNN
import pickle
from utils import get_longest_shot, pic_norm

with open("./data/HMDB_search_split.pkl", "rb") as file:
    search_split = pickle.load(file)
with open("./data/HMDB_corpus_split.pkl", "rb") as file:
    corpus_split = pickle.load(file)

def get_full_path(base_path, entry):
    return base_path + "/" + entry[1] + "/" + entry[2]


def cosine_similarity(input_embeddings, target_embeddings):
    scores = []
    for input_embedding, target_embedding in zip(input_embeddings, target_embeddings):
        scores.append(torch.cosine_similarity(input_embedding, target_embedding, 0).item())
    return sum(scores) / len(scores)

def calc_embeddings(model, video):
    model.eval()
    embeddings = []
    for path in video:
        img = pic_norm(path, (96, 72))
        with torch.no_grad():
            x = torch.unsqueeze(torch.Tensor(img), 0).to("cuda")
            _, embedding = model(x)
            embeddings.append(embedding.detach().squeeze())
    return embeddings

def save_embeddings(model):
    base_path = "./data/HMDB_main_frame"
    embedding_dict = {}
    for entry in corpus_split:
        video = get_longest_shot(get_full_path(base_path, entry))
        embeddings = calc_embeddings(model, video)
        embedding_dict[entry[1] + "/" + entry[2]] = embeddings
    for entry in search_split:
        video = get_longest_shot(get_full_path(base_path, entry))
        embeddings = calc_embeddings(model, video)
        embedding_dict[entry[1] + "/" + entry[2]] = embeddings
    with open("./data/embeddings.pkl", "wb") as file:
        pickle.dump(embedding_dict, file)


def calc_score(input_video, target_video):
    input_embeddings = embeddings_dict[input_video]
    target_embeddings = embeddings_dict[target_video]

    return cosine_similarity(input_embeddings, target_embeddings)


def search_video(video_name):
    base_path = "./data/HMDB_main_frame"
    input_video = video_name[0] + "/" + video_name[1]

    scores = []
    for entry in corpus_split:
        target_video = entry[1] + "/" +entry[2]
        scores.append((
            calc_score(input_video, target_video),
            entry
        ))

    scores = sorted(scores, key=lambda x: x[0], reverse=True)[:5]
    return scores

    #for score in scores:
        #print("score : {}   type : {}   name : {}".format(score[0], score[1][1], score[1][2]))

if __name__ == "__main__":
    model = CNN()
    model.load_state_dict(torch.load("./data/model.pt"))
    model.to("cuda")
    save_embeddings(model)
else:
    with open("./data/embeddings.pkl", "rb") as file:
        embeddings_dict = pickle.load(file)