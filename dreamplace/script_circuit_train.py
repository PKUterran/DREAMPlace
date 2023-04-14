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
from GenerateParamJson import create_param_json,creat_param


from thirdparty.TexasCyclone.data.utils import check_dir
from thirdparty.TexasCyclone.train.argument import parse_pretrain_args
from thirdparty.TexasCyclone.train.pretrain_ours import pretrain_ours
from typing import List
import json
import logging

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
        generate_data(netlist_dir,params,save_type,for_test=False)

def generate_circuit_netlist_names(circuit_dir,train=False):
    file_names = os.listdir(circuit_dir)
    netlist_names = []
    for file_name in file_names:
        if train:
            if int(file_name.split('-')[0]) > 10179:
                continue
        netlist_names.append(os.path.join(circuit_dir,file_name))
    return netlist_names

def generate_circuit_param_json(circuit_dir,train=False):
    file_names = os.listdir(circuit_dir)
    param_jsons = []
    for file_name in file_names:
        if train:
            if int(file_name.split('-')[0]) > 10179:
                continue
        param_json_path = os.path.join(circuit_dir,file_name,"param.json")
        param_jsons.append(param_json_path)
        param = creat_param()
        param["lef_input"] = "/root/circuitnet.lef"
        param["def_input"] = os.path.join(circuit_dir,file_name,file_name)
        json_data = json.dumps(param).replace(", ",",\r    ")
        with open(param_json_path,"w") as f:
            f.write(json_data)
    return param_jsons


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    train_netlist_names = generate_circuit_netlist_names(circuit_dir='/root/autodl-tmp/circuitnet/DEF',train=True)
    valid_netlist_names = train_netlist_names
    test_netlist_names = train_netlist_names
    train_param_json_list = generate_circuit_param_json(circuit_dir='/root/autodl-tmp/circuitnet/DEF',train=True)
    valid_param_json_list = train_param_json_list
    test_param_json_list = train_param_json_list
    generate_data_list(train_netlist_names,train_param_json_list,1)
    generate_data_list(valid_netlist_names,valid_param_json_list,1)
    generate_data_list(test_netlist_names,test_param_json_list,1)
    ############Pretrain
    check_dir(LOG_DIR)
    check_dir(FIG_DIR)
    check_dir(MODEL_DIR)
    args = parse_pretrain_args()
    args.lr = 5e-5
    name = args.name
    args.name = "pretrain_"+name
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
    args.lr = 1e-6
    args.epochs=10
    args.name = "train_"+name
    args.model = "pretrain_"+name
    placer = GNNPlace(raw_cell_feats, raw_net_feats, raw_pin_feats, config,args)
    # placer.load_dict(f"./model/{args.model}.pkl",device)
    # placer.load_dict(f"./model/pretrain_cellflow_kahypar_cellprop.pkl",device)
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