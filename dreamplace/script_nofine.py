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
import pandas as pd

LOG_DIR = 'log/pretrain'
FIG_DIR = 'visualize/pretrain'
MODEL_DIR = 'model'
NETLIST_DIR='benchmarks'
PARAM_DIR='test/OurModel_lowepoch'

def create_time_json_file(file_name:str):
    if os.path.exists(os.path.join("/root/DREAMPlace/time",file_name)):
        return
    data = {}
    with open(os.path.join("/root/DREAMPlace/time",file_name),"w") as f:
        f.write(json.dumps(data))

def add_netlist_name2time_json_file(file_name:str,netlist_name:str):
    with open(os.path.join("/root/DREAMPlace/time",file_name),"r") as f:
        data = json.load(f)
    if netlist_name in data:
        return
    data[netlist_name] = -1
    with open(os.path.join("/root/DREAMPlace/time",file_name),"w") as f:
        f.write(json.dumps(data))

def generate_data_list(netlist_dir_list:List[str],param_dir_list:List[str]):
    # create_time_json_file("cellflow_time.json")
    # create_time_json_file("grouping_time.json")
    # create_time_json_file("overlap_ratio.json")

    
    for netlist_dir,param_dir in zip(netlist_dir_list,param_dir_list):

        netlist_name = netlist_dir.split("/")[-1]
        # add_netlist_name2time_json_file("cellflow_time.json",netlist_name)
        # add_netlist_name2time_json_file("grouping_time.json",netlist_name)
        # add_netlist_name2time_json_file("overlap_ratio.json",netlist_name)
        
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
        # f'{PARAM_DIR}/mgc_fft_1/mgc_fft_1.json',
        # f'{PARAM_DIR}/ispd19_test1/ispd19_test1.json',
        # f'{PARAM_DIR}/mgc_superblue19/mgc_superblue19.json',
        # f'{PARAM_DIR}/mgc_des_perf_1.json',
        # f'{PARAM_DIR}/mgc_fft_1.json',
        # f'{PARAM_DIR}/mgc_fft_2.json',
        # f'{PARAM_DIR}/mgc_fft_a.json',
        # f'{PARAM_DIR}/mgc_fft_b.json',
        # f'{PARAM_DIR}/mgc_matrix_mult_1.json',
        # f'{PARAM_DIR}/mgc_matrix_mult_2.json',
        # f'{PARAM_DIR}/mgc_matrix_mult_a.json',
        # f'{PARAM_DIR}/mgc_superblue12.json',
        # f'{PARAM_DIR}/mgc_superblue14.json',
        # f'{PARAM_DIR}/mgc_superblue19.json',
        # 'test/OurModel_lowepoch/mgc_fft_1.json',
        # 'test/OurModel_lowepoch/mgc_fft_2.json',
        # 'test/OurModel_lowepoch/mgc_fft_a.json',
        # 'test/OurModel_lowepoch/mgc_fft_b.json',
        # 'test/OurModel_lowepoch/mgc_fft_b.json',
        # 'test/OurModel_lowepoch/mgc_fft_b.json',
        # 'test/OurModel_lowepoch/mgc_fft_b.json',
        f'{PARAM_DIR}/ispd19_test1.json',
        f'{PARAM_DIR}/ispd19_test2.json',
        f'{PARAM_DIR}/ispd19_test3.json',
        # # f'{PARAM_DIR}/ispd19_test4.json',
        f'{PARAM_DIR}/ispd19_test6.json',
        f'{PARAM_DIR}/ispd19_test7.json',
        f'{PARAM_DIR}/ispd19_test8.json',
        f'{PARAM_DIR}/ispd19_test9.json',
        f'{PARAM_DIR}/ispd19_test10.json',
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
        f'{NETLIST_DIR}/ispd2019/ispd19_test1',
        f'{NETLIST_DIR}/ispd2019/ispd19_test2',
        f'{NETLIST_DIR}/ispd2019/ispd19_test3',
        # # f'{NETLIST_DIR}/ispd2019/ispd19_test4',
        f'{NETLIST_DIR}/ispd2019/ispd19_test6',
        f'{NETLIST_DIR}/ispd2019/ispd19_test7',
        f'{NETLIST_DIR}/ispd2019/ispd19_test8',
        f'{NETLIST_DIR}/ispd2019/ispd19_test9',
        f'{NETLIST_DIR}/ispd2019/ispd19_test10',
    ]
    ############Train
    generate_data_list(test_netlist_names,test_param_json_list)
    result = {}

    os.environ["OMP_NUM_THREADS"] = "%d" % (16)
    args = parse_train_args()
    jump_model_inference = False
    jump_dreamplace = True
    if not jump_model_inference:
        test_placedb_list = load_placedb(test_param_json_list,test_netlist_names,'test',1)
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
        placer.logs.append({'epoch':0})
        # for placedb,netlist_name in zip(test_placedb_list,test_netlist_names):
        #     netlist_name = netlist_name.split('/')[-1]
        #     _ = placer.evaluate_place(placedb,placedb.netlist,netlist_name,use_tqdm=False)
        # placer.logs = [{'epoch':0}]
        for placedb,netlist_name,param_dir in zip(test_placedb_list,test_netlist_names,test_param_json_list):
            netlist_name = netlist_name.split('/')[-1]
            params = Params.Params()
            params.load(param_dir)
            __placedb = PlaceDB.PlaceDB()
            __placedb(params)
            metric_dict = placer.evaluate_place(placedb,placedb.netlist,netlist_name,detail_placement=True,use_tqdm=True)
            placer.logs[-1].update(metric_dict)
            result[netlist_name] = {"eval time":placer.logs[-1][f"{netlist_name} eval_time"],"hpwl":placer.logs[-1]["hpwl"] / params.scale_factor}
        print(result)
        with open(f"./result/{args.name}/result.json","w") as f:
            for k,v in result.items():
                for k_,v_ in v.items():
                    v[k_] = float(v_)
                result[k] = v
            f.write(json.dumps(result))
            keys = list(result.keys())
            result_name_list = list(result[keys[0]].keys())
            df = pd.DataFrame()
            df['netlist'] = keys
            for result_name in result_name_list:
                tmp_result = []
                for key in keys:
                    if not result_name in result[key]:
                        tmp_result.append(-1)
                    else:
                        tmp_result.append(float(result[key][result_name]))
                df[result_name] = tmp_result
            df.to_excel(f"./result/{args.name}/result.xlsx",index=False)