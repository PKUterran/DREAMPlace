import os
import sys
sys.path.append(os.path.join(os.path.abspath('.'),'build'))
sys.path.append(os.path.abspath('.'))
sys.path.append('./thirdparty/TexasCyclone')


from thirdparty.TexasCyclone.data.utils import check_dir
from thirdparty.TexasCyclone.train.argument import parse_pretrain_args
from thirdparty.TexasCyclone.train.pretrain_ours import pretrain_ours

LOG_DIR = 'log/pretrain'
FIG_DIR = 'visualize/pretrain'
MODEL_DIR = 'model'
NETLIST_DIR='../benchmarks'
train_datasets = [
    # '../Placement-datasets/dac2012/superblue2',
    # '../Placement-datasets/dac2012/superblue3',
    # '../Placement-datasets/dac2012/superblue6',
    # 'data/test/dataset1/large',
    f'{NETLIST_DIR}/ispd2015/mgc_fft_1',
    f'{NETLIST_DIR}/ispd2015/mgc_fft_2',
    f'{NETLIST_DIR}/ispd2015/mgc_fft_a',
    f'{NETLIST_DIR}/ispd2015/mgc_fft_b',
    # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_1',
    # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_2',
    # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_a',
    # f'{NETLIST_DIR}/ispd2015/mgc_superblue19',
]
valid_datasets = [
    # '../Placement-datasets/dac2012/superblue2',
    # 'data/test/dataset1/large-noclu',
    f'{NETLIST_DIR}/ispd2015/mgc_fft_1',
    f'{NETLIST_DIR}/ispd2015/mgc_fft_2',
    f'{NETLIST_DIR}/ispd2015/mgc_fft_a',
    f'{NETLIST_DIR}/ispd2015/mgc_fft_b',
    # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_1',
    # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_2',
    # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_a',
    # f'{NETLIST_DIR}/ispd2015/mgc_superblue19',
    # f'{NETLIST_DIR}/ispd2015/mgc_superblue19',

]
test_datasets = [
    # '../Placement-datasets/dac2012/superblue2',
    # 'data/test/dataset1/small',
    f'{NETLIST_DIR}/ispd2015/mgc_fft_1',
    f'{NETLIST_DIR}/ispd2015/mgc_fft_2',
    f'{NETLIST_DIR}/ispd2015/mgc_fft_a',
    f'{NETLIST_DIR}/ispd2015/mgc_fft_b',
    # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_1',
    # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_2',
    # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_a',
    # f'{NETLIST_DIR}/ispd2015/mgc_superblue19',
    # f'{NETLIST_DIR}/ispd2015/mgc_superblue19',
    # f'{NETLIST_DIR}/ispd2015/mgc_superblue19',

]

if __name__ == '__main__':
    check_dir(LOG_DIR)
    check_dir(FIG_DIR)
    check_dir(MODEL_DIR)
    args = parse_pretrain_args()
    pretrain_ours(
        args=args,
        train_datasets=train_datasets,
        valid_datasets=valid_datasets,
        test_datasets=test_datasets,
        log_dir=LOG_DIR,
        fig_dir=FIG_DIR,
        model_dir=MODEL_DIR
    )
