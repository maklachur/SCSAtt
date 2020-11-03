
class Config(object):

    batch_size = 32
    num_workers= 0 
    epoch_num  = 50
    show_step  = 1
    
    #setup the training dataset path carefully
    GOT_10k_dataset_directory = 'I:\\TEST\\datata_test'
    
    #setup the testing dataset path carefully
    OTB_dataset_directoty = 'I:\\Benchmark\\otb-100'

config = Config()
