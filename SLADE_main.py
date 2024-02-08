import math
import logging
import time
import sys
import random
import argparse
from pathlib import Path
import pickle

import torch
import numpy as np
import copy

from model.SLADE_TGN import SLADE_TGN
from utils.utils import get_neighbor_finder
from utils.data_processing import get_data_node_classification
from evaluation.evaluation import eval_anomaly_node_detection
import pdb
import pandas as pd
from tqdm import tqdm




if __name__ == '__main__':

    ### Argument and global variables
    parser = argparse.ArgumentParser('dynamic contrastive anomaly detection')
    parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia / reddit / email-eu_testinj / digg_testinj / uci_testinj)',
                        default='wikipedia')
    parser.add_argument('--bs', type=int, default=100, help='Batch_size')
    parser.add_argument('--n_degree', type=int, default=20, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-6, help='Learning rate')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--message_dim', type=int, default=128, help='Dimensions of the messages')
    parser.add_argument('--memory_dim', type=int, default=256, help='Dimensions of the memory for '
                                                                    'each user')
    parser.add_argument('--agg_type', type=str, default='TGAT', help='Aggregation type for memory recovery (only TGAT)')
    parser.add_argument('--negative_memory_type', type=str, default='train', help='Negative memory type: train/random')
    parser.add_argument('--message_updater', type=str, default='mlp', help='message updater type: mlp/identity')
    parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn"], help='Type of memory updater')
    parser.add_argument('--training_ratio', type=float, default=0.85, help='training data ratio')
    parser.add_argument('--lr_decay', type=float, default=0.8, help='learning rate decay ratio')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay ratio')
    
    parser.add_argument('--srf', type=float, default=0.1, help='weight of source memory reconstruction contrastive loss')
    parser.add_argument('--drf', type=float, default=0.1, help='weight of destination memory reconstruction contrastive loss')
    
    parser.add_argument('--only_drift_loss_score', action='store_true', help = 'using drift loss and score only')
    parser.add_argument('--only_recovery_loss_score', action='store_true', help = 'using recovery loss and score only')
    parser.add_argument('--only_drift_score', action='store_true', help = 'using drift score and both loss')
    parser.add_argument('--only_rec_score', action='store_true', help = 'using recovery score and both loss')
  
    parser.add_argument('--test_inference_time', action='store_true', help = 'Test with inference time')
    
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    DATA = args.data
    NUM_LAYER = 1 # only 1 hop neighbors aggregation possible
    LEARNING_RATE = args.lr
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim
    memory_agg_type = args.agg_type
    negative_memory_type = args.negative_memory_type
    message_updater = args.message_updater

    Path("log/").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('log/{}srf{}_drf{}_epoch{}_lr{}_bs{}_memdim{}_msgdim{}_{}.log'.format(DATA, args.srf,args.drf, NUM_EPOCH, LEARNING_RATE, BATCH_SIZE, MEMORY_DIM, MESSAGE_DIM, str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)   
    
    logger.info(args)
    logger.info('Data Processing')
    full_data, train_data, test_data = \
        get_data_node_classification(DATA, training_ratio=args.training_ratio) 

    

    max_idx = max(full_data.unique_nodes)

    train_ngh_finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_idx)
    full_ngh_finder = get_neighbor_finder(full_data, uniform=False, max_node_idx=max_idx)

    src_neighbors, _, src_neighbors_time = full_ngh_finder.get_temporal_neighbor_tqdm(full_data.sources, full_data.timestamps, NUM_NEIGHBORS)
    dst_neighbors, _, dst_neighbors_time = full_ngh_finder.get_temporal_neighbor_tqdm(full_data.destinations, full_data.timestamps, NUM_NEIGHBORS)

    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    test_aucs = []
    epoch_times = []
    for i in range(args.n_runs):
        logger.info('Dynamic anomaly detection start - runs: {}'.format(str(i)))
        dcl_tgn = SLADE_TGN(neighbor_finder=train_ngh_finder, n_nodes=full_data.n_unique_nodes, n_edges=full_data.n_interactions,
                                 device=device,n_layers=NUM_LAYER, n_heads=NUM_HEADS,
                                dropout=DROP_OUT, message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM, n_neighbors=NUM_NEIGHBORS,
                                memory_agg_type=memory_agg_type, negative_memory_type=negative_memory_type, message_updater=message_updater,
                                memory_updater=args.memory_updater, src_reg_factor=args.srf, dst_reg_factor=args.drf,
                                only_drift_loss=args.only_drift_loss_score, only_recovery_loss = args.only_recovery_loss_score)
        
        dcl_tgn = dcl_tgn.to(device)

        train_data_sources = torch.from_numpy(train_data.sources).long().to(device)
        train_data_destinations = torch.from_numpy(train_data.destinations).long().to(device)
        train_data_timestamps = torch.from_numpy(train_data.timestamps).float().to(device)
        full_data_src_neighbors = torch.from_numpy(src_neighbors).long().to(device)
        full_data_dst_neighbors = torch.from_numpy(dst_neighbors).long().to(device)
        full_data_src_neighbors_time = torch.from_numpy(src_neighbors_time).long().to(device)
        full_data_dst_neighbors_time = torch.from_numpy(dst_neighbors_time).long().to(device)
        
        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / BATCH_SIZE)

        optimizer = torch.optim.Adam(dcl_tgn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

        time_list = []
        train_time_list = []
        negative_train_nodes = torch.from_numpy(np.array(list(set(train_data.destinations)|set(train_data.sources)))).long().to(device)
        for epoch in range(NUM_EPOCH):
            dcl_tgn.memory.__init_memory__()
            dcl_tgn.set_neighbor_finder(train_ngh_finder)
            m_loss = []
            val_aucs = []

            train_start = time.time()
            for k in tqdm(range(num_batch)):
                loss = 0
                optimizer.zero_grad()
                s_idx = k * BATCH_SIZE
                e_idx = min(num_instance, s_idx + BATCH_SIZE)

                sources_batch = train_data_sources[s_idx: e_idx]
                destinations_batch = train_data_destinations[s_idx: e_idx]
                timestamps_batch = train_data_timestamps[s_idx: e_idx]
                src_neighbors_batch = full_data_src_neighbors[s_idx: e_idx]
                dst_neighbors_batch = full_data_dst_neighbors[s_idx: e_idx]
                src_neighbors_time_batch = full_data_src_neighbors_time[s_idx: e_idx]
                dst_neighbors_time_batch = full_data_dst_neighbors_time[s_idx: e_idx]

                size = len(sources_batch)

                dcl_tgn.train()
                _, _, _, _, contrastive_loss = dcl_tgn.compute_node_diff_score(sources_batch,
                                                                                destinations_batch,
                                                                                timestamps_batch,
                                                                                src_neighbors_batch, 
                                                                                dst_neighbors_batch,
                                                                                src_neighbors_time_batch,
                                                                                dst_neighbors_time_batch,
                                                                                NUM_NEIGHBORS,
                                                                                negative_train_nodes)  #train_data.n_unique_nodes
                loss += contrastive_loss
                if args.only_drift_loss_score and k==0:
                    continue
                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())
                dcl_tgn.memory.detach_memory()
            train_end = time.time()
            train_time_list.append(train_end-train_start)

            scheduler.step()
            if args.test_inference_time:
                train_ngh_finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_idx)
                dcl_tgn.set_neighbor_finder(train_ngh_finder)
            else:
                dcl_tgn.set_neighbor_finder(full_ngh_finder)

            start = time.time()
            val_auc, pred_score, pred_mask = eval_anomaly_node_detection(dcl_tgn, test_data, BATCH_SIZE,
                                            n_neighbors=NUM_NEIGHBORS, device=device, only_rec_score = args.only_rec_score or args.only_recovery_loss_score, 
                                            only_drift_score=args.only_drift_loss_score or args.only_drift_score, test_inference_time=args.test_inference_time)

            end = time.time()
            time_list.append(end-start)
            logger.info("Epoch {} - mloss: {:.4f} val auc: {:.4f}".format(str(epoch), sum(m_loss)/len(m_loss), val_auc))
            val_aucs.append(val_auc)

        test_auc = val_aucs[-1]
        logger.info("test auc: {:.4f}".format(test_auc))
        test_aucs.append(test_auc)

        train_time_np = np.array(train_time_list)
        logger.info('training time: {:.4f} std: {:.4f}'.format(np.mean(train_time_np),np.std(train_time_np)))
        time_np = np.array(time_list)
        logger.info('inference time: {:.4f} std: {:.4f}'.format(np.mean(time_np),np.std(time_np)))

    test_np = np.array(test_aucs)
    logger.info('Test end')
    logger.info('train_ratio: {:.2f}'.format(args.training_ratio))
    logger.info('final mean: {:.4f} std: {:.4f}'.format(np.mean(test_np),np.std(test_np)))

