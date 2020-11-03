from __future__ import absolute_import, print_function

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from got10k.datasets import ImageNetVID, GOT10k
from pairwise import Pairwise
from siamfc import TrackerSiamFC
from got10k.experiments import *
import numpy as np

from config import config

if __name__ == '__main__':

    # setup the desired dataset for training
    name = 'GOT-10k'
    assert name in ['VID', 'GOT-10k', 'All']
    if name == 'GOT-10k':
        seq_dataset = GOT10k(config.GOT_10k_dataset_directory, subset='train')
        pair_dataset = Pairwise(seq_dataset)
    elif name == 'VID':
        seq_dataset = ImageNetVID(config.Imagenet_dataset_directory, subset=('train', 'val'))
        pair_dataset = Pairwise(seq_dataset)
    elif name == 'All':
        seq_got_dataset = GOT10k(config.GOT_10k_dataset_directory, subset='train')
        seq_vid_dataset = ImageNetVID(config.Imagenet_dataset_directory, subset=('train', 'val'))
        pair_dataset = Pairwise(seq_got_dataset) + Pairwise(seq_vid_dataset)

    print(len(pair_dataset))

    # setup the data loader
    cuda = torch.cuda.is_available()
    loader = DataLoader(pair_dataset,
                        batch_size = config.batch_size,
                        shuffle    = True,
                        pin_memory = cuda,
                        drop_last  = True,
                        num_workers= config.num_workers)

    # setup the tracker
    #net_path = 'model/model_e32.pth' 
    tracker = TrackerSiamFC()

    for epoch in range(config.epoch_num):
        train_loss = []
        for step, batch in enumerate(tqdm(loader)):

            loss = tracker.step(batch,
                                backward=True,
                                update_lr=(step == 0))
                                
            train_loss.append(loss)
            sys.stdout.flush()

        # save the model checkpoint
        directory = 'model'
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        net_path = os.path.join('model', 'model_e%d.pth' % (epoch + 1))
        torch.save(tracker.net.state_dict(), net_path)
        print('Epoch [{}]: Loss: {:.5f}'.format( epoch + 1, np.mean(train_loss)))

