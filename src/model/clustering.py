from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn import cluster 
from sklearn.cluster import DBSCAN, KMeans
from typing import List
import json
from pprint import pprint
from tqdm import tqdm

def _encode_headlines(model, headlines: List[str]):
    embeddings = model.encode(headlines)
    # out = {}
    # for headline, embedding in zip(headlines, embeddings):
    #     out[headline] = embedding
    # print(embeddings.shape)
    
    return embeddings

def _group_to_cluster(headlines, labels):
    headline_by_cluster = {}
    
    for headline, cluster in zip(headlines, labels):
        if cluster not in headline_by_cluster:
            headline_by_cluster[cluster] = [headline]
        else:
            headline_by_cluster[cluster].append(headline)

    return headline_by_cluster

def _dbscan_clustering(headlines, embedding_array):
    for ep in np.arange(0.5, 0.8, 0.1):
        for min_sample in [3, 5, 9, 14, 21, 32, 50]:
            clustering = DBSCAN(eps=ep, metric='cosine', min_samples=min_sample).fit(embedding_array)
            if len(set(clustering.labels_)) != 1:
                print(ep, min_sample, len(set(clustering.labels_)))
                print(clustering.labels_)
                print('================')
            # headline_by_cluster = _group_to_cluster(headlines, clustering.labels_)
            # for cluster, hlines in headline_by_cluster.items():
            #     print(cluster)
            #     print(len(hlines))
            #     print(hlines[:5])

def _kmeans_clustering(headlines, embedding_array):
    clustering = KMeans(n_clusters=4, random_state=69420).fit(embedding_array)
    
    return clustering.labels_
    
    # for n_cluster in range(2, 10, 1):
    #     clustering = KMeans(n_clusters=n_cluster).fit(embedding_array)
    #     print('=================', n_cluster)
        # headline_by_cluster = _group_to_cluster(headlines, clustering.labels_)
        # cluster_count = {k : len(v) for k, v in headline_by_cluster.items()}
        # print(cluster_count)
        # for cluster, hlines in headline_by_cluster.items():
        #         print(cluster, hlines[:5])
        
def _merge_data_to_final():
    with open('data\headline_per_date.json', 'r') as f:
        headlines = json.load(f)
    
    with open('data\stockprice_per_date.json', 'r') as f:
        stocks = json.load(f)
    
    out = {}
    
    for i_sample, sample in enumerate(stocks):
        if i_sample == 15: break
        out_sample = {}
        date = sample['formatted_time']
        headline_for_date = headlines.get(date, [])
        
        out_sample['time'] = date
        # out_sample['headlines'] = 
    

if __name__ == '__main__':
    print('hello world')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    data_out = {}
    
    with open('data\headline_per_date.json', 'r') as f:
        data = json.load(f)
    
    with open('data\stockprice_per_date.json', 'r') as f:
        date_ = json.load(f)
        
    dates = [i['formatted_time'] for i in date_]
    assert len(dates) == len(date_)
    print(len(dates))
    
    for i_sample, (date, sample) in enumerate(tqdm(data.items())):
        # if i_sample == 5: break
        if date in dates and len(sample) > 4:
            sample_out = []
            daily_embedding = _encode_headlines(model, sample)
            embedding = np.stack(daily_embedding, axis=0)
            cluster_labels = _kmeans_clustering(None, embedding)
            for headline, emb, cluster in zip(sample, daily_embedding, cluster_labels):
                sample_out.append({'headline': headline, 'emb': emb.tolist(), 'cluster': int(cluster)})
            data_out[date] = sample_out
    
    print(len(data_out))
    
    # with open('data\headline_with_cluster_per_filtered_date.json', 'w') as f:
    #     json.dump(data_out, f)
    
    aggregated_data = {}
    for date, headlines in data_out.items():
        cluster_embeddings = {}
        for headline in headlines:            
            if headline['cluster'] not in cluster_embeddings:
                cluster_embeddings[headline['cluster']] = [np.asarray(headline['emb'], dtype=np.float32)]
            else:
                cluster_embeddings[headline['cluster']].append(np.asarray(headline['emb'], dtype=np.float32))
        
        daily_embedding = {clus: np.mean(np.stack(embs, axis=0), axis=0).tolist() for clus, embs in cluster_embeddings.items()}    
        assert len(daily_embedding) == 4
        aggregated_data[date] = daily_embedding
    
    assert len(data_out) == len(aggregated_data)
        
    with open('data\\aggregated_headline_by_date.json', 'w') as f:
        json.dump(aggregated_data, f)
    
        
        
    

    