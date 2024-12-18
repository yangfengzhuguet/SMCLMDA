import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--isfuse', type=str, default='yes', help='Whether to test using converged multiple views')
    parser.add_argument('--isuse', type=str, default='no', help='Whether to use the fuse function')
    parser.add_argument('--parent_dir', type=str, default='data/HMDD v_4/predata', help='The parent_dir of os.path')
    parser.add_argument('--parent_dir_', type=str, default='data/HMDD v_4/10-fold', help='The parent_dir_sub of os.path')

    parser.add_argument('--mi_number', type=int, default=535, help='The number of miRNA')
    parser.add_argument('--di_number', type=int, default=302, help='The number of disease')
    parser.add_argument('--pos_sum', type=int, default=10, help='The number of meta-path postive sample pairs')
    parser.add_argument('--lambda_2', type=float, default=0.4, help='The parameter in the total loss')

    # GConv
    parser.add_argument('--fm', type=int, default=64, help='Hidden  layer of miRNA in graph convolution') # 246:-1、+1
    parser.add_argument('--fd', type=int, default=64, help='Hidden  layer of miRNA in graph convolution') # 270：-1、+1


    # GConv-meta
    parser.add_argument('--meta_inchannels', type=int, default=64, help='input layer of meta-path graph convolution') # 64
    parser.add_argument('--meta_outchannels', type=int, default=64, help='output layer of meta-path graph convolution')# 64

    # construct-learning
    parser.add_argument('--cl_hidden', type=int, default=64, help='Hidden layer of feature project in the CL')# 64
    parser.add_argument('--temperature', type=float, default=0.7, help='The temperature in the CL loss function')
    parser.add_argument('--lambda_1', type=float, default=0.5, help='The parameter in the CL loss function')

    # MLP
    parser.add_argument('--in_feat', type=int, default=64, help='Input layer of mlp')  # 64
    parser.add_argument('--out_feat', type=int, default=64, help='Output layer of mlp') # 64

    # SLA
    parser.add_argument('--sla_hidden', type=int, default=64, help='Input layer of SLAattention') # 64
    parser.add_argument('--sla_dropout', type=float, default=0.5, help='Dropout of SLAattention') # 0.1


    #
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate')
    parser.add_argument('--we_decay', type=float, default=1e-5, help='The weight decay')
    parser.add_argument('--epoch', type=int, default=300, help='The train epoch')

    return parser.parse_args()
