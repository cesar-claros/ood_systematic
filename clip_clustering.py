#%%
import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
#%%
dict_clip = {'cifar10':'cifar10', 'cifar100':'cifar100', 'supercifar':'supercifar100','tiny-imagenet-200':'tinyimagenet',
            'super_cifar100':'supercifar100','cifar100_modelvit_bbvit_lr0.01':'cifar100'}
metrics_show = [('global','kid mean'),('global','fid'),\
                ('class-aware','inv text alignment mean'),('class-aware','img centroid dist mean'),\
                ('group_name','')]
metrics = [('global','kid mean'),('global','fid'),('class-aware','inv text alignment mean'),('class-aware','img centroid dist mean')]

n_clusters = 3
#%%
dataset = 'tiny-imagenet-200'
# Analysis non-MCD vs MCD
# ssh://caviness/work/cniel/sw/ClipUncertainty/project/notebooks/clip_proximity_cifar10.joblib
path_clip = f'clip_scores/clip_proximity_{dict_clip[dataset]}.json'
# path_clip = f'/work/cniel/sw/ClipUncertainty/project/notebooks/clip_proximity_tinyimagenet.json'
with open(path_clip, "r") as f:
    clip_distances = json.load(f)
# metrics_dict = {'knn_mean':'knn mean','kid_mean':'kid mean','inv_text_alignment_mean':'inv text alignment mean'}
distances_df = pd.DataFrame.from_dict(clip_distances, orient='index')
#%%
dist_list = []
for col in distances_df.columns:
    dist_df = distances_df[col].apply(pd.Series)
    dist_df.columns = dist_df.columns.str.replace('_',' ')
    dist_list.append(dist_df)
distances_df = pd.concat(dist_list, keys=['global','class-aware'], axis=1)
distances_df.index = distances_df.index.str.replace('_',' ')
#%%
distances_df[('class-aware','inv text alignment mean')] = 1 - distances_df[('class-aware','text alignment mean')]
distances_df.rename(index={f'{dict_clip[dataset]}': 'test'}, inplace=True)
test_row = distances_df.loc[['test']]
test_row['cluster_id'] = -1
test_row['group'] = '0'
distances_df = distances_df.drop('test',axis='rows')

# metrics = [('global','kid mean'),('global','fid'),('class-aware','img centroid dist mean')]
distances_metrics_df = distances_df[metrics].values

X = StandardScaler().fit_transform(distances_metrics_df)
# X = distances_df['global'][['fid','kid_mean']].values
# 3 clusters in multi-metric space
km = KMeans(n_clusters=n_clusters, n_init=50, random_state=0).fit(X)
distances_df["cluster_id"] = km.labels_
cluster_order = (
    distances_df.groupby("cluster_id")[[('global','fid')]]
    .mean()
    .sort_values(by=('global','fid'))
    .index.to_list()
)
id_to_name = {cluster_order[0]:"1", cluster_order[1]:"2", cluster_order[2]:"3",}
id_to_name_dist = {cluster_order[0]:"Near", cluster_order[1]:"Mid", cluster_order[2]:"Far",'0':"ID"}
distances_df = pd.concat([test_row,distances_df],axis=0)
distances_df["group"] = distances_df["cluster_id"].map(id_to_name)
distances_df["group_name"] = distances_df["cluster_id"].map(id_to_name_dist)
distances_df = distances_df.sort_values(by='group', ascending=True)
distances_df[metrics_show].to_csv(f'clip_distances_{dict_clip[dataset]}.csv')

# %%
print(distances_df[metrics_show]\
        .style\
        .set_caption(f"CLIP-based distance metrics. Dataset={dict_clip[dataset]}")
        .background_gradient(axis=0, cmap="coolwarm")
        .format(precision=4)\
        .to_latex(environment='table', convert_css=True, column_format='r|p{2.5cm}|p{2.5cm}|p{2.5cm}|p{2.5cm}|c'))
# %%
