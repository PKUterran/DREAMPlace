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
import json
from copy import deepcopy

LOG_DIR = 'log/pretrain'
FIG_DIR = 'visualize/pretrain'
MODEL_DIR = 'model'
NETLIST_DIR='benchmarks'

def floatRange(startInt, stopInt, stepInt, precision):
    f = []
    for x in range(startInt, stopInt, stepInt):
        f.append(x/(10**precision))
    return f

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
        # 'test/OurModel/mgc_superblue19/mgc_superblue19.json',
        # 'test/OurModel/mgc_des_perf_1.json',
        # 'test/OurModel/mgc_fft_1.json',
        # 'test/OurModel/mgc_fft_2.json',
        # 'test/OurModel/mgc_fft_a.json',
        # 'test/OurModel/mgc_fft_b.json',
        # 'test/OurModel/mgc_matrix_mult_1.json',
        # 'test/OurModel/mgc_matrix_mult_2.json',
        # 'test/OurModel/mgc_matrix_mult_a.json',
        # 'test/OurModel/mgc_superblue12.json',
        # 'test/OurModel/mgc_superblue14.json',
        # 'test/OurModel/mgc_superblue19.json',
        # 'test/OurModel/ispd19_test1.json',
        # 'test/OurModel/ispd19_test2.json',
        # 'test/OurModel/ispd19_test3.json',
        # 'test/OurModel/ispd19_test4.json',
        # 'test/OurModel/ispd19_test6.json',
        # 'test/OurModel/ispd19_test7.json',
        # 'test/OurModel/ispd19_test8.json',
        'test/OurModel/ispd19_test9.json',
        # 'test/OurModel/ispd19_test10.json',

    ]
    test_netlist_names = [
        # f'{NETLIST_DIR}/dac2012/superblue2'
        # f'{NETLIST_DIR}/ispd2015/mgc_des_perf_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_b',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue12',
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue14',
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue19',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test1',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test2',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test3',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test4',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test6',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test7',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test8',
        f'{NETLIST_DIR}/ispd2019/ispd19_test9',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test10',
    ]
    ############Train
    generate_data_list(test_netlist_names,test_param_json_list)

    os.environ["OMP_NUM_THREADS"] = "%d" % (16)
    args = parse_train_args()
    jump_model_inference = False
    if not jump_model_inference:
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
            metric_dict = placer.evaluate_place(placedb,placedb.netlist,netlist_name,detail_placement=True,use_tqdm=True)
            placer.logs[-1].update(metric_dict)
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
        result = {}
        best_hpwl = 1e12
        best_param = None
        cnt = 0
        for upper_pcof in floatRange(105,110,1,2):
            for iteration in range(180,360,40):
                for learning_rate in floatRange(1,10,3,1):
                    for density_weight_ in range(-9,-1,3):
                        for learning_rate_decay in floatRange(990,1000,2,3):
                            for gamma in range(4,15,3):
                                cnt+=1
                                density_weight = 8*(10**density_weight_)
                                params.__dict__["RePlAce_UPPER_PCOF"] = upper_pcof
                                params.__dict__["global_place_stages"][0]["iteration"] = iteration
                                params.__dict__["global_place_stages"][0]["learning_rate"] = learning_rate
                                params.__dict__["density_weight"] = density_weight
                                params.__dict__["global_place_stages"][0]["learning_rate_decay"] = learning_rate_decay
                                params.__dict__["gamma"] = gamma
                                placer = NonLinearPlace.NonLinearPlace(params, placedb)
                                metrics = placer(params, placedb)
                                param_dict = {"iteration":iteration,"learning_rate":learning_rate,"density_weight":density_weight,
                                                "learning_rate_decay":learning_rate_decay,"gamma":gamma,"RePlAce_UPPER_PCOF":upper_pcof,
                                                "dreamplace time":deepcopy(metrics[-1].optimizer_time)}
                                result[json.dumps(param_dict)] = deepcopy(metrics[-1].true_hpwl.tolist())
                                if metrics[-1].true_hpwl < best_hpwl:
                                    best_hpwl = deepcopy(metrics[-1].true_hpwl)
                                    best_param = param_dict
                                    json_data = json.dumps({"best_hpwl":best_hpwl.tolist(),"best_param":best_param})
                                    with open(f"./result/{args.name}/{netlist_name}/best_param.json","w") as f:
                                        f.write(json_data)
                                del placer
                                del metrics
                                torch.cuda.empty_cache()
        print(f"best_hpwl:{best_hpwl}")
        print(best_param)
        json_data = json.dumps(result)
        with open(f"./result/{args.name}/{netlist_name}/select_param_result.json","w") as f:
            f.write(json_data)
        json_data = json.dumps({"best_hpwl":best_hpwl.tolist(),"best_param":best_param})
        with open(f"./result/{args.name}/{netlist_name}/best_param.json","w") as f:
            f.write(json_data)
