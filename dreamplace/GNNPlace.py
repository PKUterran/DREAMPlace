import os
import sys
sys.path.append('/home/xuyanyang/RL/DREAMPlace/build')
import time
import pickle
import numpy as np
import logging
import torch
import gzip
# import copy
from copy import copy
import matplotlib.pyplot as plt

if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import BasicPlace
import PlaceObj
import NesterovAcceleratedGradientOptimizer
import EvalMetrics
import pdb
import dreamplace.ops.fence_region.fence_region as fence_region
from typing import Tuple, Dict, Any, List
sys.path.append(os.path.abspath('.'))
sys.path.append('./thirdparty/TexasCyclone')
from thirdparty.TexasCyclone.train.model.NaiveGNN import NaiveGNN
from thirdparty.TexasCyclone.train.argument import parse_train_args
from functools import reduce
import Params
from GNNPlaceDB import GNNPlaceDB
from thirdparty.TexasCyclone.data.graph import Netlist, Layout, expand_netlist, sequentialize_netlist, assemble_layout_with_netlist_info
from thirdparty.TexasCyclone.data.load_data import netlist_from_numpy_directory, layout_from_netlist_dis_angle
from tqdm import tqdm
import dgl
import torch.nn as nn
from dreamplace.ops.hpwl import hpwl
import dreamplace.ops.legality_check.legality_check as legality_check
from dreamplace.ops.rudy import rudy
import dreamplace.ops.draw_place.PlaceDrawer as PlaceDrawer
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from tqdm import tqdm






class GNNPlace(BasicPlace.BasicPlace):
    def __init__(self, params, placedb,
                raw_cell_feats: int,
                raw_net_feats: int,
                raw_pin_feats: int,
                config: Dict[str, Any]):
        """
        @brief initialization.
        @param params parameters
        @param placedb placement database
        """
        super(GNNPlace, self).__init__(params, placedb)
        self.model = NaiveGNN(raw_cell_feats, raw_net_feats, raw_pin_feats, config)
        n_param = 0
        for name, param in self.model.named_parameters():
            print(f'\t{name}: {param.shape}')
            n_param += reduce(lambda x, y: x * y, param.shape)
        print(f'# of parameters: {n_param}')
    
    def load_dict(self,
                dir_name: str,
                device: torch.device):
        if dir_name:
            print(f'\tUsing model {dir_name}')
            model_dicts = torch.load(f'{dir_name}', map_location=device)
            self.model.load_state_dict(model_dicts)
            self.model.eval()
    
    def evaluate_place(self,placedb,netlist: Netlist, netlist_name: str, use_tqdm=True, verbose=True):
        """
        本部分和我们代码的evaluate基本一样
        """
        self.model.eval()
        evaluate_cell_pos_corner_dict = {}
        print(f'\tFor {netlist_name}:')
        dict_netlist = expand_netlist(netlist)
        iter_i_sub_netlist = tqdm(dict_netlist.items(), total=len(dict_netlist.items()), leave=False) \
            if use_tqdm else dict_netlist.items()
        total_len = len(dict_netlist.items())
        dni: Dict[int, Dict[str, Any]] = {}  # dict_netlist_info

        batch_netlist_id = []
        total_batch_nodes_num = 0
        total_batch_edge_idx = 0
        batch_cell_feature = []
        batch_net_feature = []
        batch_pin_feature = []
        sub_netlist_feature_idrange = []
        batch_cell_size = []
        total_dis = []
        total_angle = []
        cnt = 0

        for nid, sub_netlist in iter_i_sub_netlist:
            dni[nid] = {}
            batch_netlist_id.append(nid)
            father, _ = sub_netlist.graph.edges(etype='points-to')
            edge_idx_num = father.size(0)
            sub_netlist_feature_idrange.append([total_batch_edge_idx, total_batch_edge_idx + edge_idx_num])
            total_batch_edge_idx += edge_idx_num
            total_batch_nodes_num += sub_netlist.graph.num_nodes('cell')
            batch_cell_feature.append(sub_netlist.cell_prop_dict['feat'])
            batch_net_feature.append(sub_netlist.net_prop_dict['feat'])
            batch_pin_feature.append(sub_netlist.pin_prop_dict['feat'])
            batch_cell_size.append(sub_netlist.cell_prop_dict['size'])
            if total_batch_nodes_num > 10000 or cnt == total_len - 1:
                batch_cell_feature = torch.vstack(batch_cell_feature)
                batch_net_feature = torch.vstack(batch_net_feature)
                batch_pin_feature = torch.vstack(batch_pin_feature)
                batch_cell_size = torch.vstack(batch_cell_size)
                batch_graph = []
                for nid_ in batch_netlist_id:
                    netlist = dict_netlist[nid_]
                    batch_graph.append(netlist.graph)
                batch_graph = dgl.batch(batch_graph)
                batch_edge_dis, batch_edge_angle = self.model.forward(
                    batch_graph, (batch_cell_feature, batch_net_feature, batch_pin_feature),batch_cell_size)
                total_dis.append(batch_edge_dis.unsqueeze(1))
                total_angle.append(batch_edge_angle.unsqueeze(1))
                for j, nid_ in enumerate(batch_netlist_id):
                    sub_netlist_ = dict_netlist[nid_]
                    begin_idx, end_idx = sub_netlist_feature_idrange[j]
                    edge_dis, edge_angle = \
                        batch_edge_dis[begin_idx:end_idx], batch_edge_angle[begin_idx:end_idx]
                    layout, dis_loss = layout_from_netlist_dis_angle(sub_netlist_, edge_dis, edge_angle)
                    assert not torch.isnan(dis_loss)
                    assert not torch.isinf(dis_loss)
                    dni[nid_]['cell_pos'] = copy(layout.cell_pos)
                batch_netlist_id = []
                sub_netlist_feature_idrange = []
                total_batch_nodes_num = 0
                total_batch_edge_idx = 0
                batch_cell_feature = []
                batch_net_feature = []
                batch_pin_feature = []
                batch_cell_size = []
            cnt += 1
            torch.cuda.empty_cache()
        layout = assemble_layout_with_netlist_info(dni, dict_netlist, device=device)
        evaluate_cell_pos_corner_dict[netlist_name] = \
            layout.cell_pos - layout.cell_size / 2
        torch.cuda.empty_cache()
        self.evaluate_from_numpy(evaluate_cell_pos_corner_dict[netlist_name].detach().cpu().numpy(),placedb)
        pass
    
    def plot_cell(self,placedb,cell_pos,cell_size_x,cell_size_y,dir_name):
        """
        这个是我之前用来画图的，从效率上看比DREAMPlace的慢很多这里仅作保留
        """
        fig = plt.figure()
        ax = plt.subplot()
        patches = []
        for i in tqdm(range(placedb.num_physical_nodes)):
            x,y = cell_pos[i],cell_pos[i+placedb.num_nodes]
            w,h = cell_size_x[i],cell_size_y[i]
            if i < placedb.num_movable_nodes:
                color = 'red'
            else:
                color = 'orange'
            patches.append(plt.Rectangle(
            tuple((x,y)),
            float(w),
            float(h),
            fill=False, color=color
        ))
        scale_x = placedb.xh - placedb.xl
        scale_y = placedb.yh - placedb.yl
        print(placedb.xl," ",placedb.xh," ",placedb.yl," ",placedb.yh)
        ax.set_xlim(placedb.xl - 0.1 * scale_x, placedb.xh + 0.1 * scale_x)
        ax.set_ylim(placedb.yl - 0.1 * scale_y, placedb.yh + 0.1 * scale_y)
        ax.add_collection(PatchCollection(patches, match_original=True))
        plt.savefig(dir_name)

    def evaluate_from_numpy(self,tmp_cell_pos,placedb):
        """
        注明 DREAMPlace的pos和我们的有所不同，从它的代码里得知，它的工作follow的eplace
        eplace中使用了filler_node的一种虚拟节点，所以这里的pos为[(num_physical_nodes+num_filler_nodes)*2]的nn.ParameterList
        前一半为x坐标,后一半为y坐标，因此参考了DREAMPlace的init_pos(在BasicPlace中)对filler_nodes的位置进行初始化
        这里这样照着DREAMPlace进行初始化原因是先前全赋值为0会在计算hpwl的cpp代码中报段错误，为了简易起见如此做
        """
        tmp_cell_pos = torch.tensor(tmp_cell_pos).to("cuda:0")
        """
        这里放进GPU也是考虑到后面有些函数似乎只能放在GPU上跑
        但是看见实现中是有cpu部分代码的，推测是设置的时候用了GPU后面没检测tensor所在设备的原因
        """
        cell_pos = torch.zeros(placedb.num_nodes*2,device = tmp_cell_pos.device)
        """
        直接flatten出来的坐标是乱的，不是想象中的前一半为第0维后一半为第一维
        """
        cell_pos[:placedb.num_physical_nodes] = tmp_cell_pos[:,0]
        cell_pos[placedb.num_nodes:placedb.num_nodes + placedb.num_physical_nodes] = tmp_cell_pos[:,1]
        cell_pos[placedb.num_physical_nodes : placedb.num_nodes] = torch.from_numpy(np.random.uniform(
                    low=placedb.xl,
                    high=placedb.xh - placedb.node_size_x[-placedb.num_filler_nodes],
                    size=placedb.num_filler_nodes,
                ))
        cell_pos[
            placedb.num_nodes + placedb.num_physical_nodes : placedb.num_nodes * 2
        ] = torch.from_numpy(np.random.uniform(
            low=placedb.yl,
            high=placedb.yh - placedb.node_size_y[-placedb.num_filler_nodes],
            size=placedb.num_filler_nodes,
        ))

        cell_pos = nn.ParameterList([nn.Parameter(cell_pos)])
        
        # print(self.pos,cell_pos)

        cell_pos[0] = self.legalize_cell(placedb,cell_pos)
        # print(self.legalize_check(placedb,cell_pos))
        self.evaluate_hpwl(placedb,cell_pos)

        # self.plot_cell(placedb,cell_pos[0].detach().cpu().numpy(),placedb.node_size_x,placedb.node_size_y,"/home/xuyanyang/RL/DREAMPlace/dreamplace/test.png")
        self.draw_place(placedb,cell_pos)
        self.evaluate_rudy(placedb,cell_pos)
        
        # cur_metric = EvalMetrics.EvalMetrics(0)
        # cur_metric.evaluate(placedb, {"hpwl": self.op_collections.hpwl_op}, cell_pos[0])
        # print(cur_metric)
        
    
    def legalize_check(self,placedb,cell_pos):
        """
        注 这个region在superblue2中似乎是没有的 但是不确定其他芯片上会不会有
        """
        legal_check = legality_check.LegalityCheck(
            node_size_x=self.data_collections.node_size_x,
            node_size_y=self.data_collections.node_size_y,
            flat_region_boxes=self.data_collections.flat_region_boxes,
            flat_region_boxes_start=self.data_collections.flat_region_boxes_start,
            node2fence_region_map=self.data_collections.node2fence_region_map,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            scale_factor=params.scale_factor,
            num_terminals=placedb.num_terminals,
            num_movable_nodes=placedb.num_movable_nodes)
        return legal_check(cell_pos[0])

    def legalize_cell(self,placedb,cell_pos):
        legalize_op = self.build_legalization(
            params, placedb, self.data_collections, self.device)
        return legalize_op(cell_pos[0])

    
    def evaluate_hpwl(self,placedb,cell_pos):
        """
        net2pin_map 每行存储net中pin的id,flat_net2pin_map将其展为1D numpy array
        flat_net2pin_start_map 存储每个net的pin id在flat_net2pin_map开始位置长度为len(flat_net2pin_map)+1 最后一个位置为len(flat_net2pin_map)
        pin2net_map 每个pin 所在的net id
        net_weights 目前superblue2看来默认全为1
        net_mask_all 那些net算hpwl 目前superblue2是全算
        """
        custom = hpwl.HPWL(
                flat_netpin=(self.data_collections.flat_net2pin_map), 
                netpin_start=(self.data_collections.flat_net2pin_start_map),
                pin2net_map=(self.data_collections.pin2net_map), 
                net_weights=(self.data_collections.net_weights), 
                net_mask=(self.data_collections.net_mask_all), 
                algorithm='net-by-net'
                )
        hpwl_value = custom.forward(self.op_collections.pin_pos_op(cell_pos[0]))
        print(f"hpwl is {hpwl_value}")
    
    def draw_place(self,placedb,cell_pos):
        """
        pin2node_map 每个pin所在node id
        site看起来是DEF文件定义的
        bin_size_x bin_size_y 算density map划分网格用的是用(self.yh - self.yl) / self.num_bins_y算的
        self.num_bins_y为输入参数 superblue2均为1024
        num_filler_nodes 为 int(round(self.total_filler_node_area / (filler_size_x * filler_size_y)))
        total_filler_node_area = max(
                    placeable_area * params.target_density - self.total_movable_node_area, 0.0
                )
        filler_size_x 为np.mean(self.node_size_x[node_size_order[
                                                int(self.num_movable_nodes * 0.05) : 
                                                int(self.num_movable_nodes * 0.95)]
                    ]
                )
        """
        custom = PlaceDrawer.PlaceDrawer.forward(
                    cell_pos[0].detach().cpu(), 
                    torch.from_numpy(placedb.node_size_x), torch.from_numpy(placedb.node_size_y), 
                    torch.from_numpy(placedb.pin_offset_x), torch.from_numpy(placedb.pin_offset_y), 
                    torch.from_numpy(placedb.pin2node_map), 
                    placedb.xl, placedb.yl, placedb.xh, placedb.yh, 
                    placedb.site_width, placedb.row_height, 
                    placedb.bin_size_x, placedb.bin_size_y, 
                    placedb.num_movable_nodes, 
                    placedb.num_filler_nodes, 
                    "/home/xuyanyang/RL/DREAMPlace/dreamplace/test.png" # png, jpg, eps, pdf, gds 
                    )
        print(custom)
        
    def evaluate_rudy(self,placedb,cell_pos):
        """
        unit_horizontal_capacity
        unit_vertical_capacity
        输入给定
        """
        rudy_op = rudy.Rudy(netpin_start=self.data_collections.flat_net2pin_start_map,
                            flat_netpin=self.data_collections.flat_net2pin_map,
                            net_weights=self.data_collections.net_weights,
                            xl=placedb.xl,
                            xh=placedb.xh,
                            yl=placedb.yl,
                            yh=placedb.yh,
                            num_bins_x=placedb.num_bins_x,
                            num_bins_y=placedb.num_bins_y,
                            unit_horizontal_capacity=placedb.unit_horizontal_capacity,
                            unit_vertical_capacity=placedb.unit_vertical_capacity)

        result_cpu = rudy_op.forward(self.op_collections.pin_pos_op(cell_pos[0]))
        print("rudy map size ")
        print(result_cpu.size())
        print("rudy map = ", result_cpu)
        print(f"max of rudy map is {torch.max(result_cpu)}")

    
if __name__ == '__main__':
    args = parse_train_args()
    params = Params.Params()
    params.printWelcome()

    # load parameters
    params.load(args.param_json)
    logging.info("parameters = %s" % (params))
    # control numpy multithreading
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

    # run placement
    tt = time.time()
    placedb = GNNPlaceDB(params,'/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2',1)
    print(f"num nodes {placedb.num_nodes}")
    print(f"num_physical_nodes {placedb.num_physical_nodes}")
    logging.info("initialize GNN placemet database takes %.3f seconds" % (time.time() - tt))

    tt = time.time()
    
    device = torch.device(args.device)
    config = {
        'DEVICE': device,
        'CELL_FEATS': args.cell_feats,
        'NET_FEATS': args.net_feats,
        'PIN_FEATS': args.pin_feats,
        'PASS_TYPE': args.pass_type,
    }
    sample_netlist = placedb.netlist
    raw_cell_feats = sample_netlist.cell_prop_dict['feat'].shape[1]
    raw_net_feats = sample_netlist.net_prop_dict['feat'].shape[1]
    raw_pin_feats = sample_netlist.pin_prop_dict['feat'].shape[1]
    placer = GNNPlace(params, placedb,raw_cell_feats, raw_net_feats, raw_pin_feats, config)
    logging.info("non-linear placement initialization takes %.2f seconds" %
                 (time.time() - tt))
    tt = time.time()
    placer.load_dict('/home/xuyanyang/RL/TexasCyclone/model/train-naive-bidir-l1-xoverlap-smallgroup.pkl',device)
    placer.evaluate_place(placedb,placedb.netlist,'superblue2')
    # dir_name = '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2/cell_pos.npy'
    # dir_name = '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2/output-refine-train-naive-bidir-l1-xoverlap-smallgroup.npy'
    # dir_name = '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2/output-train-naive-bidir-l1-xoverlap.npy'
    # dir_name = '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2/output-train-naive-bidir-l1-xoverlap-nohire-superblue2.npy'
    # dir_name = '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2/output-train-naive-bidir-l1-xoverlap-smallgroup.npy'
    # tmp_cell_pos = np.load(f'{dir_name}')
    # placer.evaluate_from_numpy(tmp_cell_pos,placedb)