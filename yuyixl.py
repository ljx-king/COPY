import os
# 设置Hugging Face镜像源为百川智能镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from utils.data_utils import read_client_data



# 指定模型和分词器的下载路径
cache_dir = './models_cache'  # 这里可以修改为你想要的路径
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
model = AutoModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
# CIFAR-10的标签名称
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 读取每个客户端的训练集数据标签和对应数量
num_clients = 20
client_labels_count = []
for client_idx in range(num_clients):
    train_data = read_client_data('Cifar10', client_idx, is_train=True)
    labels = [y.item() for _, y in train_data]
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_count_dict = {cifar10_classes[label]: count for label, count in zip(unique_labels, counts)}
    client_labels_count.append(label_count_dict)

# 将所有的标签名称转化为对应的语义向量并保存
label_vectors = {}
for label in cifar10_classes:
    inputs = tokenizer(label, return_tensors='pt')
    outputs = model(**inputs)
    # 使用[CLS]标记的输出作为句子的表示
    label_vector = outputs.last_hidden_state[:, 0, :].detach().numpy().flatten()
    label_vectors[label] = label_vector





# 对于每个客户端计算聚类向量
client_cluster_vectors = []
for client_label_count in client_labels_count:
    total_samples = sum(client_label_count.values())
    cluster_vector = np.zeros_like(label_vectors[cifar10_classes[0]])
    for label, count in client_label_count.items():
        proportion = count / total_samples
        cluster_vector += proportion * label_vectors[label]
    client_cluster_vectors.append(cluster_vector)

# 根据所有客户端的聚类向量使用余弦相似度计算出相似度矩阵并保存
client_cluster_vectors = np.array(client_cluster_vectors)
similarity_matrix = cosine_similarity(client_cluster_vectors)

# 保存相似度矩阵
# 创建保存文件夹（如果不存在）
save_dir = 'keshihua'
os.makedirs(save_dir, exist_ok=True)
# 保存文件到指定文件夹
file_path = os.path.join(save_dir, 'Cifarsimilarity_matrix.npy')
np.save(file_path, similarity_matrix)

print("标签向量和相似度矩阵已保存。")