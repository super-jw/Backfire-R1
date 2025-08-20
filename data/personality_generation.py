import os
from transformers import AutoTokenizer, AutoModel
import torch
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from openai import OpenAI
target_model = 'mistral-7b'
N = 4

class LLMEmbeddingExtractor:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="cuda:4")
        
    def encode_text(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt").to('cuda:4')
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state[:,-1,:]#.squeeze(0)
        return embeddings

    def encode_texts(self, texts: list) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state
        return embeddings

all_personality = []
with open('/cpfs04/user/sunjingwei/code/Search-R1/data/original_data/sft_dataset.json') as f:
    original_data = json.load(f)

for data in original_data:
    data = data['messages'][0]['content']
    all_personality.append(data.split('你需要在提案评估中扮演一个聪明和积极参与的公民角色。你需要在1到10的范围内给一个给定的提案打分。')[0])

if os.path.exists(f'/cpfs04/user/sunjingwei/code/Search-R1/data/original_data/{target_model}-embedding.npy'):
    embeddings=np.load(f'/cpfs04/user/sunjingwei/code/Search-R1/data/original_data/{target_model}-embedding.npy')
else:
    # create LLMEmbeddingExtractor
    model_name = "/cpfs04/shared/AI4Good/models/mistralai/Mistral-7B-Instruct-v0.2"  # 替换成实际的模型名称
    extractor = LLMEmbeddingExtractor(model_name)

    # all_personality = all_personality[:100]
    embeddings = []
    for personality in all_personality:
        embedding = extractor.encode_text(personality)
        embeddings.append(embedding.cpu())
    embeddings = np.array(embeddings).squeeze(1)
    np.save(f'/cpfs04/user/sunjingwei/code/Search-R1/data/original_data/{target_model}-embedding.npy', embeddings)

all_color = np.array([(213,105,93),(246,218,101),(82,190,128),(93,173,226),(164,105,189),(138,112,103),(255,188,167),(72,79,152),(0,0,0),(199,163,126)])/255

tsne = TSNE(n_components=2, perplexity=50)
data_tsne = tsne.fit_transform(embeddings)
image_tsne = data_tsne

y_pred = KMeans(n_clusters=N, random_state=9).fit_predict(data_tsne)
plt.scatter(image_tsne[:, 0], image_tsne[:, 1], marker='.', c=all_color[y_pred])
plt.savefig(f'/cpfs04/user/sunjingwei/code/Search-R1/data/original_data/{target_model}-embedding.png')
personality_clusters = []
for i in range(N):
    id_personality = (y_pred == i)
    each_cluster = [all_personality[i] for i, item in enumerate(id_personality) if item]
    personality_clusters.append(each_cluster)

client = OpenAI(base_url='http://35.220.164.252:3888/v1', api_key='sk-ixMcAoleUYS5j0lzkecUVyqQ8M3pkIOPbDPWfNkfjoQJtgCT')
final_n_personality = []
for each_cluster in personality_clusters:
    content = '帮助我总结以下个性：\n'
    for i, each_cluster_each_personality in enumerate(each_cluster):
        content += f'{i+1}.'
        content += each_cluster_each_personality
    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": content}
            ],
            temperature=0.5
        )
    ans = response.choices[0].message.content.strip()
    final_n_personality.append(ans)
with open(f'/cpfs04/user/sunjingwei/code/Search-R1/data/original_data/{target_model}-personality.json', "w") as f:
    json.dump(final_n_personality, f, indent=4, ensure_ascii=False)
    




