from transformers import AutoTokenizer, AutoModel
from kmeans_pytorch import kmeans
import torch
import json

cluster_num = 5 # 10, 15, 20

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

def tokenize(s: str):
    inputs = tokenizer(s, return_tensors="pt")
    return inputs['input_ids'][0]

word_vec_dict = {}
dir_name_word_dict = {}
max_len = 0
with open("synset_words.txt", "r") as f:
    while line := f.readline():
        dir_name, words = line.split(" ", 1)
        word = words.split(",")[0].split(" ")[-1].strip()
        token = tokenize(word)
        word_vec_dict[word] = token
        dir_name_word_dict[dir_name] = word
        lens = len(list(token))
        if lens > max_len:
            max_len = lens

for k, token in list(word_vec_dict.items()):
    lens = len(list(token))
    if lens < max_len:
        padding_length = max_len - lens
        padding = torch.zeros(padding_length, dtype=torch.float32)
        padded_token = torch.cat((token, padding), dim=0)
        word_vec_dict[k] = padded_token
x = torch.stack(list(word_vec_dict.values()), dim=0)

cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=cluster_num, distance='euclidean', device=torch.device('cuda:0')
)
word_ids_dict = {}
for k, x in zip(list(word_vec_dict.keys()), cluster_ids_x):
    word_ids_dict[k] = x

cluster_dir_name = [[] for i in range(cluster_num)]
for dir_name, word in list(dir_name_word_dict.items()):
    ids = word_ids_dict[word]
    cluster_dir_name[ids].append(dir_name)

with open(f"cls_cluster_{cluster_num}.json", "w") as f:
    json.dump(cluster_dir_name, f)