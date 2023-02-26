import os
import sys
sys.path.append(os.path.join(os.path.abspath('.'),'build'))
sys.path.append(os.path.abspath('.'))
sys.path.append('./thirdparty/TexasCyclone')

from GNNPlace import GNNPlace
from GNNPlaceDB import GNNPlaceDB
from GNNPlace import load_placedb
from thirdparty.TexasCyclone.train.argument import parse_train_args
import torch

if __name__ == '__main__':
    train_param_json_list = [
        # 'test/dac2012/superblue2.json'
        'test/ispd2015/lefdef/mgc_fft_1.json',
        'test/ispd2015/lefdef/mgc_fft_2.json',
        'test/ispd2015/lefdef/mgc_fft_a.json',
        'test/ispd2015/lefdef/mgc_fft_b.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_1.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_2.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_a.json',
        # 'test/ispd2015/lefdef/mgc_superblue19.json',
    ]
    train_netlist_names = [
        # '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2'
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_1',
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_2',
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_a',
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_b',
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_matrix_mult_1',
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_matrix_mult_2',
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_matrix_mult_a',
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_superblue19',
    ]
    valid_param_json_list = [
        # 'test/dac2012/superblue2.json'
        'test/ispd2015/lefdef/mgc_fft_1.json',
        'test/ispd2015/lefdef/mgc_fft_2.json',
        'test/ispd2015/lefdef/mgc_fft_a.json',
        'test/ispd2015/lefdef/mgc_fft_b.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_1.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_2.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_a.json',
    ]
    valid_netlist_names = [
        # '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2'
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_1',
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_2',
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_a',
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_b',
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_matrix_mult_1',
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_matrix_mult_2',
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_matrix_mult_a',
    ]
    test_param_json_list = [
        # 'test/dac2012/superblue2.json'
        'test/ispd2015/lefdef/mgc_fft_1.json',
        'test/ispd2015/lefdef/mgc_fft_2.json',
        'test/ispd2015/lefdef/mgc_fft_a.json',
        'test/ispd2015/lefdef/mgc_fft_b.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_1.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_2.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_a.json',
        # 'test/ispd2015/lefdef/mgc_superblue19.json',
    ]
    test_netlist_names = [
        # '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2'
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_1',
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_2',
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_a',
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_b',
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_matrix_mult_1',
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_matrix_mult_2',
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_matrix_mult_a',
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_superblue19',
    ]
    train_placedb_list =  load_placedb(train_param_json_list,train_netlist_names,'test',2)
    valid_placedb_list = load_placedb(valid_param_json_list,valid_netlist_names,'test',1)
    test_placedb_list = load_placedb(test_param_json_list,test_netlist_names,'test',1)
    

    args = parse_train_args()
    os.environ["OMP_NUM_THREADS"] = "%d" % (16)
    device = torch.device(args.device)
    config = {
        'DEVICE': device,
        'CELL_FEATS': args.cell_feats,
        'NET_FEATS': args.net_feats,
        'PIN_FEATS': args.pin_feats,
        'PASS_TYPE': args.pass_type,
    }
    sample_netlist = train_placedb_list[0].netlist
    raw_cell_feats = sample_netlist.graph.nodes['cell'].data['feat'].shape[1]
    raw_net_feats = sample_netlist.graph.nodes['net'].data['feat'].shape[1]
    raw_pin_feats = sample_netlist.graph.edges['pinned'].data['feat'].shape[1]
    placer = GNNPlace(raw_cell_feats, raw_net_feats, raw_pin_feats, config,args)
    if args.model:
        placer.load_dict(f"./model/{args.model}.pkl",device)
    # placer.load_dict('/home/xuyanyang/RL/DREAMPlace/model/pre-default-lr1e-5-mgc_fft_1.pkl',device)
    # placer.load_dict('/home/xuyanyang/RL/DREAMPlace/model/pre-default-3layer.pkl',device)
    placer.train_epochs(args,train_placedb_list=train_placedb_list,
                        train_netlist_names=train_netlist_names,
                        valid_placedb_list=valid_placedb_list,
                        valid_netlist_names=valid_netlist_names,
                        test_placedb_list=test_placedb_list,
                        test_netlist_names=test_netlist_names)