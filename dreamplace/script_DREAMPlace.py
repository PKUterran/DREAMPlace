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

def generate_param_list(netlist_dir_list:List[str],param_dir_list:List[str]):
    for netlist_dir,param_dir in zip(netlist_dir_list,param_dir_list):
        if not os.path.exists(param_dir):
            os.system(f"mkdir -p {os.path.dirname(param_dir)}")
            create_param_json(netlist_dir,param_dir)
        params = Params.Params()
        params.load(param_dir)
        
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
        'test/DREAMPlace/ispd19_test1/ispd19_test1.json'
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
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue19',
        f'{NETLIST_DIR}/ispd2019/ispd19_test1'
    ]
    ############Train
    generate_param_list(test_netlist_names,test_param_json_list)

    result = {}
    for netlist_name,param_dir in zip(test_netlist_names,test_param_json_list):
        if not os.path.exists(param_dir):
            create_param_json(param_dir)
        params = Params.Params()
        params.load(param_dir)
        placedb = PlaceDB.PlaceDB()
        placedb(params)
        netlist_name = netlist_name.split('/')[-1]
        os.system(f"mkdir -p ./result/DREAMPlace/{netlist_name}")
        params.__dict__["save_gp_dir"] = f"./result/DREAMPlace/{netlist_name}/DREAMPlace_{netlist_name}"
        placer = NonLinearPlace.NonLinearPlace(params, placedb)
        metrics = placer(params, placedb)
        result[netlist_name] ={}
        result[netlist_name]["hpwl"] = metrics[-1].true_hpwl
        result[netlist_name]["eval time"] = metrics[-1].optimizer_time
    print(result)