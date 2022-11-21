import sys
import os
sys.path.append('/home/xuyanyang/RL/DREAMPlace/build')
import re
import math
import time
import numpy as np
import torch
import logging
import Params
import dreamplace.ops.place_io.place_io as place_io
import dreamplace.ops.fence_region.fence_region as fence_region
import pdb
from PlaceDB import PlaceDB
import pickle
import json
sys.path.append(os.path.abspath('.'))
sys.path.append('./thirdparty/TexasCyclone')
from thirdparty.TexasCyclone.data.graph.Netlist import Netlist
from thirdparty.TexasCyclone.train.argument import parse_train_args
from copy import deepcopy
import dgl

datatypes = {
        'float32' : np.float32,
        'float64' : np.float64
        }

class GNNPlaceDB(PlaceDB):
    """
    @ placement database for hierarchical GNN 
    """
    def __init__(self,
                params,
                dir_name: str,
                save_type: int = 1) -> None:
        super(GNNPlaceDB,self).__init__()
        
        """
        initialize raw place database
        """
        self.read(params)
        self.initialize(params)

        """
        read data from PlaceDB
        """
        print(f'\tLoading {dir_name}')
        self.dataset_name = dir_name
        netlist_pickle_path = f'{dir_name}/netlist.pkl'
        if save_type == 1 and os.path.exists(netlist_pickle_path):
            with open(netlist_pickle_path, 'rb') as fp:
                self.netlist = pickle.load(fp)
        else:
            """
            get cell net num
            """
            n_cell = self.num_physical_nodes
            n_net = len(self.net_names)

            """
            get cell info
            
            cell size, cell degree, cell type
            """
            cells_size = torch.tensor(np.concatenate([self.node_x.reshape((-1,1)),self.node_y.reshape((-1,1))],axis=1))
            cells_degree = torch.zeros([self.num_physical_nodes,1])
            cells_type = torch.zeros([self.num_physical_nodes,1])
            fix_node_index_list = list(self.rawdb.fixedNodeIndices())
            for i in range(self.num_physical_nodes):
                cells_size[i] = torch.tensor([self.node_size_x[i],self.node_size_y[i]])
                if i in range(self.num_movable_nodes):
                    cells_type[i] = 0
                elif i in fix_node_index_list:
                    cells_type[i] = 1
                else:
                    cells_type[i] = 2

            """
            get net info, edge info, pin info 
            inclue 
            net degree, net to cell 
            pin data(0,1 pin offset x,y 2 pin direct 0:in 1:out)
            """
            pin_net_cell = []
            pin_data = torch.zeros([len(self.pin_direct),3])
            nets_degree = torch.zeros([len(self.net_names),1])

            for i,net_pin_list in enumerate(self.net2pin_map):
                nets_degree[i] = len(net_pin_list)
                for pin_id in net_pin_list:
                    parent_node_id = self.pin2node_map[pin_id]
                    cells_degree[parent_node_id] += 1
                    pin_data[pin_id] = torch.tensor([self.pin_offset_x[pin_id],self.pin_offset_y[pin_id],int(self.pin_direct[pin_id] == b'OUTPUT')])
                    pin_net_cell.append([i,parent_node_id])
            pin_net_cell = np.array(pin_net_cell)

            cells = list(pin_net_cell[:, 1])
            nets = list(pin_net_cell[:, 0])

            """
            get cell pos
            """
            if os.path.exists(f'{dir_name}/cell_pos.npy'):
                cells_pos_corner = np.load(f'{dir_name}/cell_pos.npy')
            else:
                cells_pos_corner = np.zeros(shape=[self.num_physical_nodes, 2], dtype=np.float)
            cells_ref_pos = torch.tensor(cells_pos_corner, dtype=torch.float32) + cells_size / 2
            cells_pos = cells_ref_pos.clone()
            cells_pos[cells_type[:, 0] < 1e-5, :] = torch.nan
            
            """
            split pin info
            pin pos, pin io
            """
            pins_pos = torch.tensor(pin_data[:, [0, 1]], dtype=torch.float32)
            pins_io = torch.tensor(pin_data[:, 2], dtype=torch.float32).unsqueeze(-1)

            """
            same with load_data.py function:netlist_from_numpy_directory
            read cell cluster, layout size
            """
            if os.path.exists(f'{dir_name}/cell_clusters.json'):
                with open(f'{dir_name}/cell_clusters.json') as fp:
                    cell_clusters = json.load(fp)
            else:
                cell_clusters = None
            if os.path.exists(f'{dir_name}/layout_size.json'):
                with open(f'{dir_name}/layout_size.json') as fp:
                    layout_size = json.load(fp)
            else:
                cells_pos_up_corner = cells_ref_pos + cells_size / 2
                layout_size = tuple(map(float, torch.max(cells_pos_up_corner[cells_type[:, 0] > 0.5], dim=0)[0]))

            """
            same with load_data.py function:netlist_from_numpy_directory
            construct graph, cell/net/pin feature
            """
            graph = dgl.heterograph({
                ('cell', 'pins', 'net'): (cells, nets),
                ('net', 'pinned', 'cell'): (nets, cells),
                ('cell', 'points-to', 'cell'): ([], []),
                ('cell', 'pointed-from', 'cell'): ([], []),
            }, num_nodes_dict={'cell': n_cell, 'net': n_net})

            cells_feat = torch.cat([torch.log(cells_size), cells_degree], dim=-1)
            nets_feat = torch.cat([nets_degree], dim=-1)
            pins_feat = torch.cat([pins_pos / 1000, pins_io], dim=-1)

            cell_prop_dict = {
                'ref_pos': cells_ref_pos,
                'pos': cells_pos,
                'size': cells_size,
                'feat': cells_feat,
                'type': cells_type,
            }
            net_prop_dict = {
                'degree': nets_degree,
                'feat': nets_feat,
            }
            pin_prop_dict = {
                'pos': pins_pos,
                'io': pins_io,
                'feat': pins_feat,
            }
            self.netlist = Netlist(
                graph=graph,
                cell_prop_dict=cell_prop_dict,
                net_prop_dict=net_prop_dict,
                pin_prop_dict=pin_prop_dict,
                layout_size=layout_size,
                hierarchical=cell_clusters is not None,
                cell_clusters=cell_clusters,
                original_netlist=Netlist(
                    graph=deepcopy(graph),
                    cell_prop_dict=deepcopy(cell_prop_dict),
                    net_prop_dict=deepcopy(net_prop_dict),
                    pin_prop_dict=deepcopy(pin_prop_dict),
                    layout_size=layout_size, simple=True
                )
            )

if __name__ == '__main__':
    args = parse_train_args()
    params = Params.Params()
    params.printWelcome()
    # if len(sys.argv) == 1 or '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
    #     params.printHelp()
    #     exit()
    # elif len(sys.argv) != 2:
    #     logging.error("One input parameters in json format in required")
    #     params.printHelp()
    #     exit()

    # load parameters
    # params.load(sys.argv[1])
    params.load(args.param_json)
    logging.info("parameters = %s" % (params))
    # control numpy multithreading
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

    # run placement
    tt = time.time()
    db = GNNPlaceDB(params,'/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2',1)
    logging.info("initialize GNN placemet database takes %.3f seconds" % (time.time() - tt))