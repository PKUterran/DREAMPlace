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
from thirdparty.TexasCyclone.train.functions import AreaLoss, HPWLLoss
from tqdm import tqdm
import dgl
import json
import torch.nn as nn
from dreamplace.ops.hpwl import hpwl
from dreamplace.ops.pin_pos import pin_pos
from dreamplace.ops.weighted_average_wirelength import weighted_average_wirelength
import dreamplace.ops.legality_check.legality_check as legality_check
import dreamplace.ops.macro_legalize.macro_legalize as macro_legalize
import dreamplace.ops.greedy_legalize.greedy_legalize as greedy_legalize
import dreamplace.ops.abacus_legalize.abacus_legalize as abacus_legalize
import dreamplace.ops.electric_potential.electric_potential as electric_potential
from dreamplace.ops.rudy import rudy
import dreamplace.ops.draw_place.PlaceDrawer as PlaceDrawer
import dreamplace.ops.global_swap.global_swap as global_swap
import dreamplace.ops.k_reorder.k_reorder as k_reorder
import dreamplace.ops.independent_set_matching.independent_set_matching as independent_set_matching
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from tqdm import tqdm
from torch.autograd import Function, Variable
import argparse






class GNNPlace():
    def __init__(self,
                raw_cell_feats: int,
                raw_net_feats: int,
                raw_pin_feats: int,
                config: Dict[str, Any],
                args: argparse.Namespace,):
        """
        @brief initialization.
        @param params parameters
        @param placedb placement database
        """
        # super(GNNPlace, self).__init__(params, placedb)
        """
        GNNPlace已经和DREAMPlace的Placer分开，仅有GNNPlaceDB继承了PlaceDB用来读数据用
        """
        self.model = NaiveGNN(raw_cell_feats, raw_net_feats, raw_pin_feats, config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=(1 - args.lr_decay))
        self.device = config['DEVICE']
        self.best_metric = 1e10
        self.logs: List[Dict[str, Any]] = []
        self.save_dir = os.path.join(os.path.abspath('.'),'model',f"{args.name}.pkl")
        self.log_dir = os.path.join(os.path.abspath('.'),'log/ours',f"{args.name}.json")
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
    
    
    def build_metric(self,placedb,nid):
        op_collection = {}
        op_collection['hpwl'] = self.build_op_sub_netlist_hpwl(placedb,nid,self.device)
        # op_collection['hpwl'] = self.build_op_sub_netlist_our_hpwl(placedb,nid,self.device)
        op_collection['desitypotential'] = self.build_op_sub_netlist_desitypotential(placedb,nid,self.device)
        op_collection['area'] = self.build_op_sub_netlist_area(placedb,nid,self.device)
        return op_collection
    
    def calc_metric(self,metric,layout):
        hpwl_loss = metric['hpwl'](layout.cell_pos)
        desity_loss = metric['desitypotential'](layout.cell_pos)
        area_loss = metric['area'](layout)
        return{
            'hpwl_loss':hpwl_loss,
            'desity_loss':desity_loss,
            'area_loss':area_loss,
        }

    def collect_netlist_op(self,placedb_list):
        data_op_collection = []
        for placedb in placedb_list:
            dict_netlist = expand_netlist(placedb.netlist)
            for nid,sub_netlist in dict_netlist.items():
                if nid not in placedb.sub_netlist_info:
                    continue
                tmp_collection = {}
                tmp_collection['netlist'] = sub_netlist
                tmp_collection['metric'] = self.build_metric(placedb,nid)
                data_op_collection.append(tmp_collection)
        return data_op_collection
    
    def train_epochs(self,args:argparse.Namespace,
                    train_placedb_list,
                    train_netlist_names:List,
                    valid_placedb_list,
                    valid_netlist_names:List,
                    test_placedb_list,
                    test_netlist_names:List):
        data_op_collection = self.collect_netlist_op(train_placedb_list)
        for epoch in range(0, args.epochs + 1):
            self.logs.append({'epoch':epoch})
            t0 = time.time()
            if epoch:
                for _ in range(args.train_epoch):
                    self.train_places(args,data_op_collection)
            print(f"train_time:{time.time() - t0}")
            self.logs[-1].update({'train_time': time.time() - t0})
            t1 = time.time()
            
            # self.evaluate_places(train_placedb_list,train_netlist_names,'train')
            self.evaluate_places(valid_placedb_list,valid_netlist_names,'valid')
            # self.evaluate_places(test_placedb_list,test_netlist_names,'test')
            if self.log_dir is not None:
                with open(self.log_dir,'w+') as fp:
                    json.dump(self.logs,fp)
            
        pass

    def train_places(self,args:argparse.Namespace,
                    data_op_collection):
        """
        现在的训练方式是先将placedb中所有netlist和sub netlist拿出来build出hpwl desity loss metric和netlist一起存在data_op_collection(实现在collect_netlist_op)
        训练时直接从data_op_collection里面一个一个拿netlist和对应的metric训练
        """
        self.model.train()
        t1 = time.time()
        losses = []
        n_netlist = len(data_op_collection)
        iter_i_collection = tqdm(enumerate(data_op_collection), total=n_netlist) \
                if args.use_tqdm else enumerate(data_op_collection)
        batch_netlist = []
        batch_metric = []
        total_batch_nodes_num = 0
        total_batch_edge_idx = 0
        batch_cell_feature = []
        batch_net_feature = []
        batch_pin_feature = []
        sub_netlist_feature_idrange = []
        batch_cell_size = []
        
        for j, collection in iter_i_collection:
            netlist = collection['netlist']
            batch_netlist.append(netlist)
            batch_metric.append(collection['metric'])
            father, _ = netlist.graph.edges(etype='points-to')
            edge_idx_num = father.size(0)
            sub_netlist_feature_idrange.append([total_batch_edge_idx, total_batch_edge_idx + edge_idx_num])
            total_batch_edge_idx += edge_idx_num
            total_batch_nodes_num += netlist.graph.num_nodes('cell')
            batch_cell_feature.append(netlist.cell_prop_dict['feat'])
            batch_net_feature.append(netlist.net_prop_dict['feat'])
            batch_pin_feature.append(netlist.pin_prop_dict['feat'])
            batch_cell_size.append(netlist.cell_prop_dict['size'])
            if total_batch_nodes_num > 50000 or j == n_netlist - 1:
                batch_cell_feature = torch.vstack(batch_cell_feature)
                batch_net_feature = torch.vstack(batch_net_feature)
                batch_pin_feature = torch.vstack(batch_pin_feature)
                batch_graph = dgl.batch([sub_netlist.graph for sub_netlist in batch_netlist])
                batch_cell_size = torch.vstack(batch_cell_size)
                batch_edge_dis, batch_edge_angle = self.model.forward(
                    batch_graph, (batch_cell_feature, batch_net_feature, batch_pin_feature),batch_cell_size)
                # batch_edge_dis,batch_edge_angle = batch_edge_dis.cpu(),batch_edge_angle.cpu()
                for nid, sub_netlist in enumerate(batch_netlist):
                    begin_idx, end_idx = sub_netlist_feature_idrange[nid]
                    edge_dis, edge_angle = batch_edge_dis[begin_idx:end_idx], batch_edge_angle[begin_idx:end_idx]
                    assert not torch.any(torch.isnan(edge_dis))
                    assert not torch.any(torch.isinf(edge_dis))
                    assert not torch.any(torch.isnan(edge_angle))
                    assert not torch.any(torch.isinf(edge_angle))
                    layout, dis_loss = layout_from_netlist_dis_angle(sub_netlist, edge_dis, edge_angle)
                    assert not torch.isnan(dis_loss), f"{dis_loss}"
                    assert not torch.isinf(dis_loss), f"{dis_loss}"
                    assert not torch.any(torch.isnan(layout.cell_pos))
                    assert not torch.any(torch.isinf(layout.cell_pos))
                    metric = batch_metric[nid]
                    loss_dict = self.calc_metric(metric,layout)
                    loss = sum((
                        # args.dis_lambda * dis_loss,
                        1e0 * loss_dict['hpwl_loss'],
                        1e2 * loss_dict['desity_loss'],
                        # args.overlap_lambda * loss_dict['overlap_loss'],
                        # 1e-1 * loss_dict['area_loss'],
                        # args.hpwl_lambda * loss_dict['hpwl_loss'],
                        # args.cong_lambda * loss_dict['cong_loss'],
                    ))
                    assert not torch.isinf(loss_dict['hpwl_loss']), f"{loss_dict['hpwl_loss']}"
                    assert not torch.isnan(loss_dict['hpwl_loss']), f"{loss_dict['hpwl_loss']}"
                    assert not torch.isinf(loss_dict['desity_loss']), f"{loss_dict['desity_loss']}"
                    assert not torch.isnan(loss_dict['desity_loss']), f"{loss_dict['desity_loss']}"
                    # print(f"\t\t HPWL Loss:{loss_dict['hpwl_loss']}")
                    # print(f"\t\t desitypotential Loss:{loss_dict['desity_loss']}")
                    # print(f'\t\t Discrepancy Loss: {dis_loss}')
                    # print(f"\t\t Discrepancy Loss: {loss_dict['area_loss']}")
                    losses.append(loss)
                # self.optimizer.zero_grad()
                (sum(losses) / len(losses)).backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
                self.optimizer.step()
                losses.clear()
                batch_netlist = []
                sub_netlist_feature_idrange = []
                total_batch_nodes_num = 0
                total_batch_edge_idx = 0
                batch_cell_feature = []
                batch_net_feature = []
                batch_pin_feature = []
                batch_cell_size = []
                batch_metric = []
                torch.cuda.empty_cache()
        print(f"\tTraining time per epoch: {time.time() - t1}")


        pass
    
    def build_op_sub_netlist_area(self,placedb,nid,device):
        area_op = AreaLoss()
        def build_op_area(layout):
            return area_op(layout,limit=[0,0,
                                        *placedb.sub_netlist_info[nid]['layout_size']])
        return build_op_area
        pass


    def build_op_sub_netlist_desitypotential(self,placedb,nid,device):
        # device = pos.device
        target_density = torch.empty(1,dtype=torch.float32,device=device)
        target_density.data.fill_(0.9)
        xl,xh = placedb.sub_netlist_info[nid]['xl'],placedb.sub_netlist_info[nid]['xh']
        yl,yh = placedb.sub_netlist_info[nid]['yl'],placedb.sub_netlist_info[nid]['yh']
        num_bins_x = placedb.sub_netlist_info[nid]['num_bins_x']
        num_bins_y = placedb.sub_netlist_info[nid]['num_bins_y']
        num_cells = placedb.sub_netlist_info[nid]['num_physical_nodes']
        """
        ElectricPotential里面的node要这样排
        [cell_type=0(moveable node), cell_type>1(terminal)
        所以这里对node_size_x node_size_y cell_type以及后面的pos要进行重排
        """
        node_size_x = placedb.sub_netlist_info[nid]['node_size_x'].squeeze()
        node_size_y = placedb.sub_netlist_info[nid]['node_size_y'].squeeze()
        cell_type = placedb.sub_netlist_info[nid]['cell_type'].squeeze()
        zero_index = torch.argwhere(cell_type == 0)
        one_index = torch.argwhere(cell_type > 0)
        node_size_x = torch.vstack([node_size_x[zero_index],node_size_x[one_index]]).squeeze()
        node_size_y = torch.vstack([node_size_y[zero_index],node_size_y[one_index]]).squeeze()
        cell_type = torch.vstack([cell_type[zero_index],cell_type[one_index]]).squeeze()

        bin_size_x = placedb.sub_netlist_info[nid]['bin_size_x']
        bin_size_y = placedb.sub_netlist_info[nid]['bin_size_y']

        # cell_type = placedb.sub_netlist_info[nid]['cell_type']
        num_moveable_cell = sum(cell_type == 0)
        moveable_size_x = node_size_x[:num_moveable_cell]
        _, sorted_node_map = torch.sort(moveable_size_x)
        sorted_node_map = sorted_node_map.to(torch.int32).to(device)
        node_areas = node_size_x * node_size_y
        mean_area = node_areas[:num_moveable_cell].mean().mul_(10)
        row_height = node_size_y[:num_moveable_cell].min().mul_(2)
        moveable_macro_mask = ((node_areas[:num_moveable_cell] > mean_area) & \
            (node_size_y[:num_moveable_cell] > row_height)).to(device)
        """
        注：这个moveable_macro_mask需要有fence_regions
        现在fence_regions还是None，从py调用看来DREAMPlace的superblue2是不需要
        """
        
        def bin_center_x_padded(xh_,xl_,padding,num_bins_x):
            bin_size_x = (xh_ - xl_) / num_bins_x
            xl = xl_ - padding * bin_size_x
            xh = xh_ + padding * bin_size_x
            bin_center_x = torch.from_numpy(
                placedb.bin_centers(xl, xh, bin_size_x)).to(device)
            return bin_center_x
        def bin_center_y_padded(yh_,yl_,padding,num_bins_y):
            bin_size_y = (yh_ - yl_) / num_bins_y
            yl = yl_ - padding * bin_size_y
            yh = yh_ + padding * bin_size_y
            bin_center_y = torch.from_numpy(
                placedb.bin_centers(yl, yh, bin_size_y)).to(device)
            return bin_center_y
        
        
        electric_potential_op = electric_potential.ElectricPotential(
            node_size_x=node_size_x.to(device),
            node_size_y=node_size_y.to(device),
            bin_center_x=bin_center_x_padded(xl,xh, 0, num_bins_x),#此处函数实现直接抄DREAMPlace
            bin_center_y=bin_center_y_padded(yl,yh, 0, num_bins_y),
            target_density=torch.tensor(target_density,requires_grad=False).to(device),
            xl=float(xl),
            yl=float(yl),
            xh=float(xh),
            yh=float(yh),
            bin_size_x=float(bin_size_x),
            bin_size_y=float(bin_size_y),
            num_movable_nodes=num_moveable_cell,
            num_terminals=num_cells - num_moveable_cell,
            num_filler_nodes=0,
            padding=0,
            deterministic_flag=False,
            sorted_node_map=sorted_node_map,
            movable_macro_mask=moveable_macro_mask,
            fast_mode=False,
            region_id=None,
            fence_regions=None,
            node2fence_region_map=None,
            placedb=None)
        def build_op_desitypotential(pos_):
            """
            注：此处将pos变换从以0,0为中心点变为以(width / 2, height / 2)为中心点
            相当于左下角点变为(0,0)
            """
            pos = pos_.clone()
            pos = pos.reshape([-1])
            pos = torch.concat([pos[:num_cells][zero_index],pos[:num_cells][one_index],pos[num_cells:][zero_index],pos[num_cells:][one_index]])
            pos = torch.hstack([pos[:num_cells],pos[:num_cells]])
            pos += torch.tensor([placedb.sub_netlist_info[nid]['width']/2.0,placedb.sub_netlist_info[nid]['height']/2.0],device=device)
            # pos_var = Variable(pos.reshape([-1]),requires_grad=True).to(device)
            pos_var = pos.reshape([-1]).to(device)
            return electric_potential_op(pos_var) / placedb.sub_netlist_info[nid]['num_nets']
        return build_op_desitypotential

    def build_op_sub_netlist_our_hpwl(self,placedb,nid,device):
        area_op = HPWLLoss(device)
        def build_op_area(layout):
            return area_op(layout)
        return build_op_area

    def build_op_sub_netlist_hpwl(self,placedb,nid,device):
        """
        构建sub netlist hpwl loss
        """
        # device = pos.device
        bin_size_x = (placedb.sub_netlist_info[nid]['xh'] - placedb.sub_netlist_info[nid]['xl']) / placedb.sub_netlist_info[nid]['num_bins_x']
        bin_size_y = (placedb.sub_netlist_info[nid]['yh'] - placedb.sub_netlist_info[nid]['yl']) / placedb.sub_netlist_info[nid]['num_bins_y']
        gamma = torch.tensor(1e1 * 4.0 * (bin_size_x + bin_size_y))#此处参数参考DREAMPlace的初始参数赋值，DREAMPlace的gamma会随着训练变化，这里先不变，
        #此外DREAMPlace在代码注释意思是gamma越小和真实的hpwl越接近？
        pin_pos_op = pin_pos.PinPos(
            pin_offset_x=torch.tensor(placedb.sub_netlist_info[nid]['pin_offset_x']).to(device),
            pin_offset_y=torch.tensor(placedb.sub_netlist_info[nid]['pin_offset_y']).to(device),
            pin2node_map=torch.tensor(placedb.sub_netlist_info[nid]['pin2node_map'],dtype=torch.long).to(device),
            flat_node2pin_map=torch.tensor(placedb.sub_netlist_info[nid]['flat_node2pin_map'],dtype=torch.int32).to(device),
            flat_node2pin_start_map=torch.tensor(placedb.sub_netlist_info[nid]['flat_node2pin_start_map'],dtype=torch.int32).to(device),
            num_physical_nodes=placedb.sub_netlist_info[nid]['num_physical_nodes'],
            algorithm="node-by-node"
        )
        hpwl_op = weighted_average_wirelength.WeightedAverageWirelength(
            flat_netpin=Variable(torch.tensor(placedb.sub_netlist_info[nid]['flat_net2pin_map'],dtype=torch.int32)).to(device),
            netpin_start=Variable(torch.tensor(placedb.sub_netlist_info[nid]['flat_net2pin_start_map'],dtype=torch.int32)).to(device),
            pin2net_map=torch.tensor(placedb.sub_netlist_info[nid]['pin2net_map'],dtype=torch.int32).to(device),
            net_weights=torch.tensor(placedb.sub_netlist_info[nid]['net_weights'],dtype=torch.float32).to(device),
            net_mask=torch.tensor(placedb.sub_netlist_info[nid]['net_mask_all'],dtype=torch.uint8).to(device),
            pin_mask=torch.tensor(placedb.sub_netlist_info[nid]['pin_mask_ignore_fixed_macros'],dtype=torch.bool).to(device),
            gamma=gamma.to(device),
            algorithm='merged'
        )
        
        def build_hpwl_op(pos_):
            pos = pos_.clone()
            # pos += torch.tensor([placedb.sub_netlist_info[nid]['width']/2.0,placedb.sub_netlist_info[nid]['height']/2.0],device=device)
            # pos_var = Variable(pos.reshape([-1]),requires_grad=True)
            pos_var = pos.reshape([-1]).to(device)
            hpwl_loss = hpwl_op(pin_pos_op(pos_var)) / placedb.sub_netlist_info[nid]['num_nets']#这里考虑想将hpwl压缩一下
            return hpwl_loss
        return build_hpwl_op
    
    def evaluate_places(self,placedb_list,netlist_names,name='train'):
        t0 = time.time()
        ds = []
        metric = 0
        for placedb,netlist_name in zip(placedb_list,netlist_names):
            metric_dict = self.evaluate_place(placedb,placedb.netlist,netlist_name,use_tqdm=True)
            metric += metric_dict['hpwl']
            d = {
                f'{netlist_name}_hpwl':float(metric_dict['hpwl']),
                f'{netlist_name}_rudy':float(metric_dict['rudy']),
            }
            self.logs[-1].update(d)
        metric /= len(netlist_names)
        print(f"inference {name} time {time.time() - t0}")
        if name == 'valid':
            if metric < self.best_metric:
                self.best_metric = metric
                print(f"Saving model to {self.save_dir}")
                torch.save(self.model.state_dict(),self.save_dir)
        



    def evaluate_place(self,placedb,netlist: Netlist, netlist_name: str, use_tqdm=True, verbose=True):
        """
        本部分和我们代码的evaluate基本一样
        """
        self.model.train()
        evaluate_cell_pos_corner_dict = {}
        print("-------")
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
        t0 = time.time()
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
            if total_batch_nodes_num > 100000 or cnt == total_len - 1:
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
                # batch_edge_dis,batch_edge_angle = batch_edge_dis.cpu(),batch_edge_angle.cpu()
                for j, nid_ in enumerate(batch_netlist_id):
                    sub_netlist_ = dict_netlist[nid_]
                    begin_idx, end_idx = sub_netlist_feature_idrange[j]
                    edge_dis, edge_angle = \
                        batch_edge_dis[begin_idx:end_idx], batch_edge_angle[begin_idx:end_idx]
                    layout, dis_loss = layout_from_netlist_dis_angle(sub_netlist_, edge_dis, edge_angle)
                    
                    # hpwl_loss = self.build_op_sub_netlist_hpwl(placedb,nid_,self.device)(layout.cell_pos)
                    # density_loss = self.build_op_sub_netlist_desitypotential(placedb,nid_,self.device)(layout.cell_pos)
                    # hpwl_loss.backward()
                    # density_loss.backward()

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
        layout = assemble_layout_with_netlist_info(dni, dict_netlist, device=self.device)
        evaluate_cell_pos_corner_dict[netlist_name] = \
            layout.cell_pos.cpu() - layout.cell_size / 2
        evaluate_cell_pos_corner_dict[netlist_name] = torch.maximum(torch.zeros_like(layout.cell_pos.cpu()),evaluate_cell_pos_corner_dict[netlist_name])
        evaluate_cell_pos_corner_dict[netlist_name] = torch.minimum(torch.tensor([placedb.xh,placedb.yh]).view([1,2]) - layout.cell_size,evaluate_cell_pos_corner_dict[netlist_name])
        """
        防止cell出界
        LG阶段没法处理出界
        """
        torch.cuda.empty_cache()
        # print(f"layout process time {time.time()-t0}")
        self.logs[-1].update({f'{netlist_name} eval_time': time.time() - t0})
        evaluate_result_dict = self.evaluate_from_numpy(evaluate_cell_pos_corner_dict[netlist_name].detach().cpu().numpy(),placedb)#evaluate_cell_pos_corner_dict[netlist_name].detach().cpu().numpy()
        for key,value in evaluate_result_dict.items():
            print(f"{key}: {value}")
        print("-------")
        return evaluate_result_dict
        
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
        tmp_cell_pos = torch.tensor(tmp_cell_pos).cpu()
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

        # cell_pos[0] = self.legalize_cell(placedb,cell_pos[0])
        # cell_pos[0] = self.detail_placement(placedb,cell_pos[0])
        # print(self.legalize_check(placedb,cell_pos[0]))
        hpwl = self.evaluate_hpwl(placedb,cell_pos[0])

        # self.plot_cell(placedb,cell_pos[0].detach().cpu().numpy(),placedb.node_size_x,placedb.node_size_y,"/home/xuyanyang/RL/DREAMPlace/dreamplace/test_plot_cell.png")
        self.draw_place(placedb,cell_pos[0])
        rudy = self.evaluate_rudy(placedb,cell_pos[0])
        
        return {'hpwl' : hpwl,
                'rudy' : rudy}
        
    
    def legalize_check(self,placedb,cell_pos):
        """
        注 这个region在superblue2中似乎是没有的 但是不确定其他芯片上会不会有
        """
        device = cell_pos.device
        legal_check = legality_check.LegalityCheck(
            node_size_x=torch.tensor(placedb.node_size_x,device=device),
            node_size_y=torch.tensor(placedb.node_size_y,device=device),
            flat_region_boxes=torch.tensor(placedb.flat_region_boxes,device=device),
            flat_region_boxes_start=torch.tensor(placedb.flat_region_boxes_start,device=device),
            node2fence_region_map=torch.tensor(placedb.node2fence_region_map,device=device),
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            scale_factor=placedb.params.scale_factor,
            num_terminals=placedb.num_terminals,
            num_movable_nodes=placedb.num_movable_nodes)
        return legal_check(cell_pos)

    def legalize_cell(self,placedb,cell_pos):
        """
        此处实现参考basicplace
        """
        device = cell_pos.device
        ml = macro_legalize.MacroLegalize(
            node_size_x=torch.from_numpy(placedb.node_size_x).to(device).to(torch.float32),
            node_size_y=torch.from_numpy(placedb.node_size_y).to(device).to(torch.float32),
            node_weights=torch.from_numpy(placedb.num_pins_in_nodes).to(device).to(torch.float32),
            flat_region_boxes=torch.from_numpy(placedb.flat_region_boxes).to(device).to(torch.float32),
            flat_region_boxes_start=torch.from_numpy(placedb.flat_region_boxes_start).to(device).to(torch.int),
            node2fence_region_map=torch.from_numpy(placedb.node2fence_region_map).to(device).to(torch.int),
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes)
        legalize_alg = greedy_legalize.GreedyLegalize
        gl = legalize_alg(
            node_size_x=torch.from_numpy(placedb.node_size_x).to(device).to(torch.float32),
            node_size_y=torch.from_numpy(placedb.node_size_y).to(device).to(torch.float32),
            node_weights=torch.from_numpy(placedb.num_pins_in_nodes).to(device).to(torch.float32),
            flat_region_boxes=torch.from_numpy(placedb.flat_region_boxes).to(device).to(torch.float32),
            flat_region_boxes_start=torch.from_numpy(placedb.flat_region_boxes_start).to(device).to(torch.int),
            node2fence_region_map=torch.from_numpy(placedb.node2fence_region_map).to(device).to(torch.int),
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=1,
            num_bins_y=64,
            #num_bins_x=64, num_bins_y=64,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes)
        # for standard cell legalization
        al = abacus_legalize.AbacusLegalize(
            node_size_x=torch.from_numpy(placedb.node_size_x).to(device).to(torch.float32),
            node_size_y=torch.from_numpy(placedb.node_size_y).to(device).to(torch.float32),
            node_weights=torch.from_numpy(placedb.num_pins_in_nodes).to(device).to(torch.float32),
            flat_region_boxes=torch.from_numpy(placedb.flat_region_boxes).to(device).to(torch.float32),
            flat_region_boxes_start=torch.from_numpy(placedb.flat_region_boxes_start).to(device).to(torch.int),
            node2fence_region_map=torch.from_numpy(placedb.node2fence_region_map).to(device).to(torch.int),
            xl=float(placedb.xl),
            yl=float(placedb.yl),
            xh=float(placedb.xh),
            yh=float(placedb.yh),
            site_width=float(placedb.site_width),
            row_height=float(placedb.row_height),
            num_bins_x=1,
            num_bins_y=64,
            #num_bins_x=64, num_bins_y=64,
            num_movable_nodes=int(placedb.num_movable_nodes),
            num_terminal_NIs=int(placedb.num_terminal_NIs),
            num_filler_nodes=int(placedb.num_filler_nodes))
        def build_legalization_op(pos):
            logging.info("Start legalization")
            pos1 = ml(pos, pos)
            pos2 = gl(pos1, pos1)
            legal = self.legalize_check(placedb,pos2)
            if not legal:
                logging.error("legality check failed in greedy legalization")
                return pos2
            return al(pos1, pos2)

        return build_legalization_op(cell_pos)
        pass
        # legalize_op = self.build_legalization(
        #     params, placedb, self.data_collections, self.device)
        # return legalize_op(cell_pos[0])

    
    def evaluate_hpwl(self,placedb,cell_pos):
        """
        net2pin_map 每行存储net中pin的id,flat_net2pin_map将其展为1D numpy array
        flat_net2pin_start_map 存储每个net的pin id在flat_net2pin_map开始位置长度为len(flat_net2pin_map)+1 最后一个位置为len(flat_net2pin_map)
        pin2net_map 每个pin 所在的net id
        net_weights 目前superblue2看来默认全为1
        net_mask_all 那些net算hpwl 目前superblue2是全算
        """
        # custom = hpwl.HPWL(
        #         flat_netpin=(self.data_collections.flat_net2pin_map), 
        #         netpin_start=(self.data_collections.flat_net2pin_start_map),
        #         pin2net_map=(self.data_collections.pin2net_map), 
        #         net_weights=(self.data_collections.net_weights), 
        #         net_mask=(self.data_collections.net_mask_all), 
        #         algorithm='net-by-net'
        #         )
        # hpwl_value = custom.forward(self.op_collections.pin_pos_op(cell_pos[0]))
        device = cell_pos.device
        pin_pos_op = pin_pos.PinPos(
            pin_offset_x=torch.tensor(placedb.pin_offset_x).to(device),
            pin_offset_y=torch.tensor(placedb.pin_offset_y).to(device),
            pin2node_map=torch.tensor(placedb.pin2node_map,dtype=torch.int32).to(device),
            flat_node2pin_map=torch.tensor(placedb.flat_node2pin_map,dtype=torch.int32).to(device),
            flat_node2pin_start_map=torch.tensor(placedb.flat_node2pin_start_map,dtype=torch.int32).to(device),
            num_physical_nodes=placedb.num_physical_nodes,
            algorithm="node-by-node"
        )
        hpwl_op = hpwl.HPWL(
            flat_netpin=torch.tensor(placedb.flat_net2pin_map,dtype=torch.int32).to(device),
            netpin_start=torch.tensor(placedb.flat_net2pin_start_map,dtype=torch.int32).to(device),
            pin2net_map=torch.tensor(placedb.pin2net_map,dtype=torch.int32).to(device),
            net_weights=torch.tensor(placedb.net_weights,dtype=torch.float32).to(device),
            net_mask=torch.from_numpy(np.ones(placedb.num_nets,dtype=np.uint8)).to(device),
            algorithm='net-by-net'
        )
        hpwl_value = hpwl_op(pin_pos_op(cell_pos)).cpu().clone().detach().data
        # print(f"hpwl is {hpwl_value}")
        return hpwl_value
    
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
                    cell_pos.detach().cpu(), 
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
        # print(custom)
        
    def evaluate_rudy(self,placedb,cell_pos):
        """
        unit_horizontal_capacity
        unit_vertical_capacity
        输入给定
        """
        device = cell_pos.device
        pin_pos_op = pin_pos.PinPos(
            pin_offset_x=torch.tensor(placedb.pin_offset_x).to(device),
            pin_offset_y=torch.tensor(placedb.pin_offset_y).to(device),
            pin2node_map=torch.tensor(placedb.pin2node_map,dtype=torch.int32).to(device),
            flat_node2pin_map=torch.tensor(placedb.flat_node2pin_map,dtype=torch.int32).to(device),
            flat_node2pin_start_map=torch.tensor(placedb.flat_node2pin_start_map,dtype=torch.int32).to(device),
            num_physical_nodes=placedb.num_physical_nodes,
            algorithm="node-by-node"
        )
        rudy_op = rudy.Rudy(netpin_start=torch.tensor(placedb.flat_net2pin_start_map).to(device),#self.data_collections
                            flat_netpin=torch.tensor(placedb.flat_net2pin_map).to(device),#self.data_collections
                            net_weights=torch.tensor(placedb.net_weights).to(device),#self.data_collections
                            xl=placedb.xl,
                            xh=placedb.xh,
                            yl=placedb.yl,
                            yh=placedb.yh,
                            num_bins_x=placedb.num_bins_x,
                            num_bins_y=placedb.num_bins_y,
                            unit_horizontal_capacity=placedb.unit_horizontal_capacity,
                            unit_vertical_capacity=placedb.unit_vertical_capacity)

        result_cpu = rudy_op.forward(pin_pos_op(cell_pos))#self.op_collections.pin_pos_op(cell_pos[0].cuda())
        result_cpu = result_cpu.cpu().clone().detach().data
        return torch.max(result_cpu)
    

    def detail_placement(self,placedb,cell_pos):
        """
        参考BasicPlace实现
        将max iter调到50，因为目前我们的GNN还是散不开cell所以要多进行DP来缓解LG带来的过大HPWL
        """
        device = cell_pos.device
        self.net_mask_all = torch.from_numpy(
                np.ones(placedb.num_nets,
                        dtype=np.uint8)).to(device)  # all nets included
        net_degrees = np.array(
            [len(net2pin) for net2pin in placedb.net2pin_map])
        net_mask = np.logical_and(
            2 <= net_degrees,
            net_degrees < 100).astype(np.uint8)
        self.net_mask_ignore_large_degrees = torch.from_numpy(net_mask).to(
            device)
        gs = global_swap.GlobalSwap(
            node_size_x=torch.from_numpy(placedb.node_size_x).to(device),
            node_size_y=torch.from_numpy(placedb.node_size_y).to(device),
            flat_region_boxes=torch.from_numpy(placedb.flat_region_boxes).to(device),
            flat_region_boxes_start=torch.from_numpy(placedb.flat_region_boxes_start).to(device),
            node2fence_region_map=torch.from_numpy(placedb.node2fence_region_map).to(device),
            flat_net2pin_map=torch.from_numpy(placedb.flat_net2pin_map).to(device),
            flat_net2pin_start_map=torch.from_numpy(placedb.flat_net2pin_start_map).to(device),
            pin2net_map=torch.from_numpy(placedb.pin2net_map).to(device),
            flat_node2pin_map=torch.from_numpy(placedb.flat_node2pin_map).to(device),
            flat_node2pin_start_map=torch.from_numpy(placedb.flat_node2pin_start_map).to(device),
            pin2node_map=torch.from_numpy(placedb.pin2node_map).to(device),
            pin_offset_x=torch.from_numpy(placedb.pin_offset_x).to(device),
            pin_offset_y=torch.from_numpy(placedb.pin_offset_y).to(device),
            net_mask=self.net_mask_ignore_large_degrees,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=placedb.num_bins_x // 2,
            num_bins_y=placedb.num_bins_y // 2,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
            batch_size=256,
            max_iters=50,
            algorithm='concurrent')
        kr = k_reorder.KReorder(
            node_size_x=torch.from_numpy(placedb.node_size_x).to(device),
            node_size_y=torch.from_numpy(placedb.node_size_y).to(device),
            flat_region_boxes=torch.from_numpy(placedb.flat_region_boxes).to(device),
            flat_region_boxes_start=torch.from_numpy(placedb.flat_region_boxes_start).to(device),
            node2fence_region_map=torch.from_numpy(placedb.node2fence_region_map).to(device),
            flat_net2pin_map=torch.from_numpy(placedb.flat_net2pin_map).to(device),
            flat_net2pin_start_map=torch.from_numpy(placedb.flat_net2pin_start_map).to(device),
            pin2net_map=torch.from_numpy(placedb.pin2net_map).to(device),
            flat_node2pin_map=torch.from_numpy(placedb.flat_node2pin_map).to(device),
            flat_node2pin_start_map=torch.from_numpy(placedb.flat_node2pin_start_map).to(device),
            pin2node_map=torch.from_numpy(placedb.pin2node_map).to(device),
            pin_offset_x=torch.from_numpy(placedb.pin_offset_x).to(device),
            pin_offset_y=torch.from_numpy(placedb.pin_offset_y).to(device),
            net_mask=self.net_mask_ignore_large_degrees,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
            K=4,
            max_iters=50)
        ism = independent_set_matching.IndependentSetMatching(
            node_size_x=torch.from_numpy(placedb.node_size_x).to(device),
            node_size_y=torch.from_numpy(placedb.node_size_y).to(device),
            flat_region_boxes=torch.from_numpy(placedb.flat_region_boxes).to(device),
            flat_region_boxes_start=torch.from_numpy(placedb.flat_region_boxes_start).to(device),
            node2fence_region_map=torch.from_numpy(placedb.node2fence_region_map).to(device),
            flat_net2pin_map=torch.from_numpy(placedb.flat_net2pin_map).to(device),
            flat_net2pin_start_map=torch.from_numpy(placedb.flat_net2pin_start_map).to(device),
            pin2net_map=torch.from_numpy(placedb.pin2net_map).to(device),
            flat_node2pin_map=torch.from_numpy(placedb.flat_node2pin_map).to(device),
            flat_node2pin_start_map=torch.from_numpy(placedb.flat_node2pin_start_map).to(device),
            pin2node_map=torch.from_numpy(placedb.pin2node_map).to(device),
            pin_offset_x=torch.from_numpy(placedb.pin_offset_x).to(device),
            pin_offset_y=torch.from_numpy(placedb.pin_offset_y).to(device),
            net_mask=self.net_mask_ignore_large_degrees,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
            batch_size=2048,
            set_size=128,
            max_iters=50,
            algorithm='concurrent')

        # wirelength for position
        def build_detailed_placement_op(pos):
            logging.info("Start ABCDPlace for refinement")
            pos1 = pos
            legal = self.legalize_check(placedb,pos1)
            logging.info("ABCDPlace input legal flag = %d" %
                         (legal))
            if not legal:
                return pos1

            # integer factorization to prime numbers
            def prime_factorization(num):
                lt = []
                while num != 1:
                    for i in range(2, int(num+1)):
                        if num % i == 0:  # i is a prime factor
                            lt.append(i)
                            num = num / i # get the quotient for further factorization
                            break
                return lt

            # compute the scale factor for detailed placement
            # as the algorithms prefer integer coordinate systems
            # scale_factor = params.scale_factor
            scale_factor = 1.0#superblue2直接设的1我就把下面这些注释了
            # if params.scale_factor != 1.0:
            #     inv_scale_factor = int(round(1.0 / params.scale_factor))
            #     prime_factors = prime_factorization(inv_scale_factor)
            #     target_inv_scale_factor = 1
            #     for factor in prime_factors:
            #         if factor != 2 and factor != 5:
            #             target_inv_scale_factor = inv_scale_factor
            #             break
            #     scale_factor = 1.0 / target_inv_scale_factor
            #     logging.info("Deriving from system scale factor %g (1/%d)" % (params.scale_factor, inv_scale_factor))
            #     logging.info("Use scale factor %g (1/%d) for detailed placement" % (scale_factor, target_inv_scale_factor))

            for i in range(1):
                pos1 = kr(pos1, scale_factor)
                legal = self.legalize_check(placedb,pos1)
                logging.info("K-Reorder legal flag = %d" % (legal))
                if not legal:
                    return pos1
                pos1 = ism(pos1, scale_factor)
                legal = self.legalize_check(placedb,pos1)
                logging.info("Independent set matching legal flag = %d" %
                             (legal))
                if not legal:
                    return pos1
                pos1 = gs(pos1, scale_factor)
                legal = self.legalize_check(placedb,pos1)
                logging.info("Global swap legal flag = %d" % (legal))
                if not legal:
                    return pos1
                pos1 = kr(pos1, scale_factor)
                legal = self.legalize_check(placedb,pos1)
                logging.info("K-Reorder legal flag = %d" % (legal))
                if not legal:
                    return pos1
            return pos1

        return build_detailed_placement_op(cell_pos)

def load_placedb(params_json_list,
                    netlist_names,
                    name):
    """
    直接将所有placedb都进来存起来
    """
    print(f'Loading {name} data...')
    placedb_list = []
    for param_json, netlist_name in zip(params_json_list,netlist_names):
        params = Params.Params()
        # load parameters
        params.load(param_json)
        placedb = GNNPlaceDB(params,netlist_name,1)
        print(f"load {netlist_name} netlist")
        print(f"num nodes {placedb.num_nodes}")
        print(f"num_physical_nodes {placedb.num_physical_nodes}")
        placedb_list.append(placedb)
    return placedb_list


"""
后面需要一个scrip_train的文件
这里先从简了
"""
def script_train():
    train_param_json_list = [
        'test/dac2012/superblue2.json'
    ]
    train_netlist_names = [
        '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2'
    ]
    valid_param_json_list = [
        'test/dac2012/superblue7.json'
    ]
    valid_netlist_names = [
        '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue7'
    ]
    test_param_json_list = [
        'test/dac2012/superblue9.json'
    ]
    test_netlist_names = [
        '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue9'
    ]
    train_placedb_list = load_placedb(train_param_json_list,train_netlist_names,'train')
    # valid_placedb_list = load_placedb(valid_param_json_list,valid_netlist_names,'valid')
    # test_placedb_list = load_placedb(test_param_json_list,test_netlist_names,'test')
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
    raw_cell_feats = sample_netlist.cell_prop_dict['feat'].shape[1]
    raw_net_feats = sample_netlist.net_prop_dict['feat'].shape[1]
    raw_pin_feats = sample_netlist.pin_prop_dict['feat'].shape[1]
    placer = GNNPlace(raw_cell_feats, raw_net_feats, raw_pin_feats, config,args)
    placer.load_dict('/home/xuyanyang/RL/TexasCyclone/model/pre-naive-bidir-l1-xoverlap-smallgroup.pkl',device)
    # placer.load_dict('/home/xuyanyang/RL/DREAMPlace/model/onlyhpwl.pkl',device)
    # placer.train_places(args,train_placedb_list)
    placer.train_epochs(args,train_placedb_list=train_placedb_list,
                        train_netlist_names=train_netlist_names,
                        valid_placedb_list=train_placedb_list,
                        valid_netlist_names=valid_netlist_names,
                        test_placedb_list=train_placedb_list,
                        test_netlist_names=test_netlist_names)

if __name__ == '__main__':
    script_train()
    # args = parse_train_args()
    # params = Params.Params()
    # params.printWelcome()

    # # load parameters
    # params.load(args.param_json)
    # logging.info("parameters = %s" % (params))
    # # control numpy multithreading
    # os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

    # # run placement
    # tt = time.time()
    # placedb = GNNPlaceDB(params,'/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2',1)
    # print(f"num nodes {placedb.num_nodes}")
    # print(f"num_physical_nodes {placedb.num_physical_nodes}")
    # logging.info("initialize GNN placemet database takes %.3f seconds" % (time.time() - tt))

    # tt = time.time()
    
    # device = torch.device(args.device)
    # config = {
    #     'DEVICE': device,
    #     'CELL_FEATS': args.cell_feats,
    #     'NET_FEATS': args.net_feats,
    #     'PIN_FEATS': args.pin_feats,
    #     'PASS_TYPE': args.pass_type,
    # }
    # sample_netlist = placedb.netlist
    # raw_cell_feats = sample_netlist.cell_prop_dict['feat'].shape[1]
    # raw_net_feats = sample_netlist.net_prop_dict['feat'].shape[1]
    # raw_pin_feats = sample_netlist.pin_prop_dict['feat'].shape[1]
    # placer = GNNPlace(raw_cell_feats, raw_net_feats, raw_pin_feats, config, args)
    # logging.info("non-linear placement initialization takes %.2f seconds" %
    #              (time.time() - tt))
    # tt = time.time()


    # ####################
    # placer.load_dict('/home/xuyanyang/RL/TexasCyclone/model/train-naive-bidir-l1-xoverlap-smallgroup.pkl',device)
    # placer.load_dict('/home/xuyanyang/RL/DREAMPlace/model/default.pkl',device)
    # placer.load_dict('/home/xuyanyang/RL/DREAMPlace/model/1hpwl_1desity_1area.pkl',device)
    # placer.evaluate_place(placedb,placedb.netlist,'superblue2')
    ####################

    ##############
    # dir_name = '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2/cell_pos.npy'
    # dir_name = '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2/output-refine-train-naive-bidir-l1-xoverlap-smallgroup.npy'
    # dir_name = '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2/output-train-naive-bidir-l1-xoverlap.npy'
    # dir_name = '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2/output-train-naive-bidir-l1-xoverlap-nohire-superblue2.npy'
    # dir_name = '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2/output-train-naive-bidir-l1-xoverlap-smallgroup.npy'
    # tmp_cell_pos = np.load(f'{dir_name}')
    # print(placer.evaluate_from_numpy(tmp_cell_pos,placedb))
    ##################
