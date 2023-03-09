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
import PlaceDB
from dreamplace import NonLinearPlace
import logging

LOG_DIR = 'log/pretrain'
FIG_DIR = 'visualize/pretrain'
MODEL_DIR = 'model'
NETLIST_DIR='benchmarks'

def generate_data_list(netlist_dir_list:List[str],param_dir_list:List[str]):
    for netlist_dir,param_dir in zip(netlist_dir_list,param_dir_list):
        if not os.path.exists(param_dir):
            os.system(f"mkdir -p {os.path.dirname(param_dir)}")
            create_param_json(netlist_dir,param_dir,ourmodel=True)
        params = Params.Params()
        params.load(param_dir)
        generate_data(netlist_dir,params,for_test=True)
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    test_param_json_list = [
        # 'test/dac2012/superblue2.json'
        # 'test/ispd2015/lefdef/mgc_fft_1.json',
        # 'test/ispd2015/lefdef/mgc_fft_2.json',
        # 'test/ispd2015/lefdef/mgc_fft_a.json',
        # 'test/ispd2015/lefdef/mgc_fft_b.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_1.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_2.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_a.json',
        # 'test/ispd2015/lefdef/mgc_superblue19.json',
        # 'test/OurModel/mgc_fft_1/mgc_fft_1.json',
        # 'test/OurModel/ispd19_test1/ispd19_test1.json',
        'test/OurModel/mgc_superblue19/mgc_superblue19.json'
    ]
    test_netlist_names = [
        # f'{NETLIST_DIR}/dac2012/superblue2'
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_b',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_a',
        f'{NETLIST_DIR}/ispd2015/mgc_superblue19',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test1'
    ]
    ############Train
    generate_data_list(test_netlist_names,test_param_json_list)
    result = {}

    os.environ["OMP_NUM_THREADS"] = "%d" % (16)
    args = parse_train_args()
    jump_model_inference = False
    if not jump_ourmodel:
        test_placedb_list = load_placedb(test_param_json_list,test_netlist_names,'test',2)
        device = torch.device(args.device)
        config = {
            'DEVICE': device,
            'CELL_FEATS': args.cell_feats,
            'NET_FEATS': args.net_feats,
            'PIN_FEATS': args.pin_feats,
            'PASS_TYPE': args.pass_type,
        }
        sample_netlist = test_placedb_list[0].netlist
        raw_cell_feats = sample_netlist.graph.nodes['cell'].data['feat'].shape[1]
        raw_net_feats = sample_netlist.graph.nodes['net'].data['feat'].shape[1]
        raw_pin_feats = sample_netlist.graph.edges['pinned'].data['feat'].shape[1]
        placer = GNNPlace(raw_cell_feats, raw_net_feats, raw_pin_feats, config,args)
        if args.model:
            placer.load_dict(f"./model/{args.model}.pkl",device)
        placer.jump_LGDP = True
        placer.logs.append({'epoch':0})
        for placedb,netlist_name in zip(test_placedb_list,test_netlist_names):
            netlist_name = netlist_name.split('/')[-1]
            _ = placer.evaluate_place(placedb,placedb.netlist,netlist_name,use_tqdm=False)
        placer.logs = [{'epoch':0}]
        for placedb,netlist_name in zip(test_placedb_list,test_netlist_names):
            netlist_name = netlist_name.split('/')[-1]
            metric_dict = placer.evaluate_place(placedb,placedb.netlist,netlist_name,use_tqdm=True)
            placer.logs[-1].update(metric_dict)
            result[netlist_name] = {"eval time":placer.logs[-1][f"{netlist_name} eval_time"]}
    for netlist_dir,param_dir in zip(test_netlist_names,test_param_json_list):
        if not os.path.exists(param_dir):
            create_param_json(netlist_dir,param_dir)
        params = Params.Params()
        params.load(param_dir)
        placedb = PlaceDB.PlaceDB()
        placedb(params)
        netlist_name = netlist_dir.split('/')[-1]
        params.__dict__["init_pos_dir"] = f'./result/{args.name}/{netlist_name}/{netlist_name}.npy'
        params.__dict__["save_gp_dir"] = f"./result/{args.name}/{netlist_name}/{netlist_name}"
        placer = NonLinearPlace.NonLinearPlace(params, placedb)
        metrics = placer(params, placedb)
        if netlist_name not in result:
            result[netlist_name] = {"hpwl":0,"eval time":0}
        result[netlist_name]["hpwl"] = metrics[-1].true_hpwl
        result[netlist_name]["eval time"] += metrics[-1].optimizer_time
    print(result)