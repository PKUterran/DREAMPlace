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
from GenerateData import generate_data
import Params
from GenerateParamJson import create_param_json


from thirdparty.TexasCyclone.data.utils import check_dir
from thirdparty.TexasCyclone.train.argument import parse_pretrain_args
from thirdparty.TexasCyclone.train.pretrain_ours import pretrain_ours
from typing import List

LOG_DIR = 'log/pretrain'
FIG_DIR = 'visualize/pretrain'
MODEL_DIR = 'model'
NETLIST_DIR='./benchmarks'

def generate_data_list(netlist_dir_list:List[str],param_dir_list:List[str],save_type=1):
    for netlist_dir,param_dir in zip(netlist_dir_list,param_dir_list):
        if not os.path.exists(param_dir):
            create_param_json(netlist_dir,param_dir)
        params = Params.Params()
        params.load(param_dir)
        generate_data(netlist_dir,params,save_type)
        
if __name__ == '__main__':
    train_param_json_list = [
        # 'test/dac2012/superblue2.json'
        'test/ispd2015/lefdef/mgc_fft_1.json',
        # 'test/ispd2015/lefdef/mgc_fft_2.json',
        # 'test/ispd2015/lefdef/mgc_fft_a.json',
        # 'test/ispd2015/lefdef/mgc_fft_b.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_1.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_2.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_a.json',
        # 'test/ispd2015/lefdef/mgc_superblue19.json',
    ]
    train_netlist_names = [
        # f'{NETLIST_DIR}/dac2012/superblue2'
        f'{NETLIST_DIR}/ispd2015/mgc_fft_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_b',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue19',
    ]
    valid_param_json_list = [
        # 'test/dac2012/superblue2.json'
        'test/ispd2015/lefdef/mgc_fft_1.json',
        # 'test/ispd2015/lefdef/mgc_fft_2.json',
        # 'test/ispd2015/lefdef/mgc_fft_a.json',
        # 'test/ispd2015/lefdef/mgc_fft_b.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_1.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_2.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_a.json',
    ]
    valid_netlist_names = [
        # f'{NETLIST_DIR}/dac2012/superblue2'
        f'{NETLIST_DIR}/ispd2015/mgc_fft_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_b',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_a',
    ]
    test_param_json_list = [
        # 'test/dac2012/superblue2.json'
        'test/ispd2015/lefdef/mgc_fft_1.json',
        # 'test/ispd2015/lefdef/mgc_fft_2.json',
        # 'test/ispd2015/lefdef/mgc_fft_a.json',
        # 'test/ispd2015/lefdef/mgc_fft_b.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_1.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_2.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_a.json',
        # 'test/ispd2015/lefdef/mgc_superblue19.json',
    ]
    test_netlist_names = [
        # f'{NETLIST_DIR}/dac2012/superblue2'
        f'{NETLIST_DIR}/ispd2015/mgc_fft_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_b',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue19',
    ]
    generate_data_list(train_netlist_names,train_param_json_list,1)
    generate_data_list(valid_netlist_names,valid_param_json_list,1)
    generate_data_list(test_netlist_names,test_param_json_list,1)
    ############Pretrain
    check_dir(LOG_DIR)
    check_dir(FIG_DIR)
    check_dir(MODEL_DIR)
    args = parse_pretrain_args()
    pretrain_ours(
        args=args,
        train_datasets=train_netlist_names,
        valid_datasets=valid_netlist_names,
        test_datasets=test_netlist_names,
        log_dir=LOG_DIR,
        fig_dir=FIG_DIR,
        model_dir=MODEL_DIR
    )
    ############Pretrain

    ############Train
    train_placedb_list =  load_placedb(train_param_json_list,train_netlist_names,'train',2)
    valid_placedb_list = load_placedb(valid_param_json_list,valid_netlist_names,'valid',2)
    test_placedb_list = load_placedb(test_param_json_list,test_netlist_names,'test',2)

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
    # if args.model:
    #     placer.load_dict(f"./model/{args.model}.pkl",device)
    if os.path.exists(os.path.join(MODEL_DIR,f"{args.model}.pkl")):
        placer.load_dict(os.path.join(MODEL_DIR,f"{args.model}.pkl"),device)
    placer.train_epochs(args,train_placedb_list=train_placedb_list,
                        train_netlist_names=train_netlist_names,
                        valid_placedb_list=valid_placedb_list,
                        valid_netlist_names=valid_netlist_names,
                        test_placedb_list=test_placedb_list,
                        test_netlist_names=test_netlist_names)