import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import random
import pdb


def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])  # int(e[3])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list})


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df

def reindex_with_node_id_map(df, bipartite=False):
  new_df = df.copy()
  if bipartite:
    n_unique_src = len(set(new_df.u))
    n_unique_dst = len(set(new_df.i))
    node_id_map = {}
    counter = 0
    for src in new_df.u:
      if src not in node_id_map:
        node_id_map[src] = counter
        counter += 1
    #pdb.set_trace()
    assert (counter==n_unique_src)
    new_df.i = new_df.i + new_df.u.max() + 1

    for dst in new_df.i:
      if dst not in node_id_map:
        node_id_map[dst] = counter
        counter += 1
    
    new_df.u = new_df.u.apply(lambda x: node_id_map[x])
    new_df.i = new_df.i.apply(lambda x: node_id_map[x])

    assert(new_df.u.max() - new_df.u.min() + 1 == n_unique_src)
    assert(new_df.i.max() - new_df.i.min() + 1 == n_unique_dst)
    assert(new_df.u.max()+1 == new_df.i.min())

    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    n_unique_nodes = len(set(new_df.u)|set(new_df.i))
    node_id_map = {}
    counter = 0  
    for i in range(len(new_df.u)):
      if new_df.u[i] not in node_id_map:
        node_id_map[new_df.u[i]] = counter
        counter += 1

      if new_df.i[i] not in node_id_map:
        node_id_map[new_df.i[i]] = counter
        counter += 1
  
    new_df.u = new_df.u.apply(lambda x: node_id_map[x])
    new_df.i = new_df.i.apply(lambda x: node_id_map[x])
    assert (counter==n_unique_nodes)
    assert (len(set(new_df.u)|set(new_df.i))==n_unique_nodes)
    print(counter, new_df.u.max(), new_df.i.max())
    
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  return new_df  

  
  


def run(data_name, bipartite=True):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = './data/{}.csv'.format(data_name)
  OUT_DF = './data/ml_{}.csv'.format(data_name)
  OUT_FEAT = './data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

  df = preprocess(PATH)
  #new_df = reindex(df, bipartite)
  new_df = reindex_with_node_id_map(df, bipartite)
  max_idx = max(new_df.u.max(), new_df.i.max())
  zero_feat = np.zeros((max_idx + 1, 172))
  zero_edge_feat = np.zeros((len(new_df), 172))


  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, zero_edge_feat)
  np.save(OUT_NODE_FEAT, zero_feat)

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')


args = parser.parse_args()

run(args.data, bipartite=args.bipartite)