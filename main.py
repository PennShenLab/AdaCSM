import numpy as np
from utils.plotting import plot_KM, plot_Weibull_cdf
from utils.data_utils import load_data
from utils.general_utils import train_test_DCSM, test_DCSM, sample_weibull, calibration
import time
import argparse
import scipy.stats as st
import torch
from sklearn.preprocessing import StandardScaler
import pickle as pkl


def init_config():
    parser = argparse.ArgumentParser(description='Deep Clustering Survival Machines')
    # model hyper-parameters
    parser.add_argument('--dataset', type=str, default='FRAMINGHAM',
                        help='dataset in [support, flchain, PBC, FRAMINGHAM, sim]')
    parser.add_argument('--is_normalize', type=bool, default=True, help='whether to normalize data')
    parser.add_argument('--is_cluster', type=bool, default=True, help='whether to use DCSM to do clustering')
    parser.add_argument('--is_generate_sim', type=bool, default=True, help='whether we generate simulation data')
    parser.add_argument('--is_save_sim', type=bool, default=True, help='whether we save simulation data')
    parser.add_argument('--num_inst', default=200, type=int,
                        help='specifies the number of instances for simulation data')
    parser.add_argument('--num_feat', default=10, type=int,
                        help='specifies the number of features for simulation data')
    parser.add_argument('--cuda_device', default=0, type=int,
                        help='specifies the index of the cuda device')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate for optimizer')
    parser.add_argument('--layers', default='[50]', type=str, help='hidden layer sizes as string e.g. "[50]" or "[50,50]"')
    parser.add_argument('--discount', default=0.5, type=float, help='specifies number of discount parameter')
    parser.add_argument('--iters', default=2000, type=int, help='number of training iterations')
    parser.add_argument('--patience', default=100, type=int, help='patience for early stopping')
    parser.add_argument('--early_stopping', default=False, type=bool, help='whether to use early stopping')
    parser.add_argument('--weibull_shape', default=2, type=int, help='specifies the Weibull shape')
    parser.add_argument('--num_cluster', default=2, type=int, help='specifies the number of clusters')
    parser.add_argument('--train_DCSM', default=True, type=bool, help='whether to train DCSM')
    parser.add_argument('--use_moe', action='store_true', help='use Mixture of Experts instead of MLP')
    parser.add_argument('--num_experts', default=4, type=int, help='number of experts for MoE')
    parser.add_argument('--top_k', default=None, type=int, help='use only top-k experts (None = all experts)')
    # MoE hyperparameters
    parser.add_argument('--moe_dropout', default=0.0, type=float, help='dropout rate for MoE expert networks')
    parser.add_argument('--gate_dropout', default=0.0, type=float, help='dropout rate for MoE gate')
    parser.add_argument('--gate_temperature', default=1.0, type=float, help='temperature for gate softmax')
    parser.add_argument('--routing_noise_std', default=0.0, type=float, help='std of routing noise during training')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay for optimizer')
    parser.add_argument('--load_balance_lambda', default=0.0, type=float, help='load balancing loss weight for MoE')
    parser.add_argument('--progress_every', default=0, type=int, help='print progress every N epochs (0 disables)')

    args = parser.parse_args()
    parser.print_help()
    return args


start_time = time.perf_counter()
print('start time is: ', start_time)

args = init_config()  # input params from command
torch.cuda.set_device(args.cuda_device)  # set cuda device
data_name = args.dataset

result_DCSM = []
logrank_DCSM = []
rae_nc_DCSM_list = []
rae_c_DCSM_list = []
cal_DCSM_list = []

# normalization
is_normalized = args.is_normalize

########################################
#      Train and Test Models
########################################

# Build param dict from CLI args
import ast
layers = ast.literal_eval(args.layers)  # Convert "[50]" string to [50] list

param = {'learning_rate': args.learning_rate, 'layers': layers, 'k': 2,
        'iters': args.iters, 'distribution': 'Weibull', 'discount': args.discount,
        'patience': args.patience, 'early_stopping': args.early_stopping}


# hold out testing with different splitting using the same parameters set
for seed in [42, 73, 666, 777, 1009]:
# for seed in [42]:
    X_train, X_test, y_train, y_test, _ = load_data(args, random_state=seed)

    print('-------------------------dataset: {}, train shape: {}, seed {}-----------------'
          .format(data_name, X_train.shape, seed))
    e_train = np.array([[item[0] * 1 for item in y_train]]).T
    t_train = np.array([[item[1] for item in y_train]]).T
    e_test = np.array([[item[0] * 1 for item in y_test]]).T
    t_test = np.array([[item[1] for item in y_test]]).T

    if is_normalized:
        print('Data are normalized')
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # X_test = scaler.fit_transform(X_test)
    else:
        print('Data are not normalized')

    if args.train_DCSM:
        print('-----------------------------train and test DCSM-------------------------------')
        model, c_index, pred_DCSM, pred_time_DCSM, rae_nc_DCSM, rae_c_DCSM \
            = train_test_DCSM(param, X_train, X_test, y_train, y_test, fix=True, method='DCSM',
                             use_moe=args.use_moe, num_experts=args.num_experts, top_k=args.top_k,
                             moe_dropout=args.moe_dropout, gate_dropout=args.gate_dropout,
                             gate_temperature=args.gate_temperature, routing_noise_std=args.routing_noise_std,
                             weight_decay=args.weight_decay, load_balance_lambda=args.load_balance_lambda,
                             progress_every=args.progress_every)
        model_suffix = '_moe' if args.use_moe else ''
        if args.top_k is not None:
            model_suffix += f'_topk{args.top_k}'
        with open('models/DCSM_{}_seed{}{}.pkl'.format(data_name, seed, model_suffix), 'wb') as file:
            pkl.dump(model, file)
    else:
        print('-----------------------------just test DCSM-------------------------------')
        with open('models/DCSM_{}_seed{}.pkl'.format(data_name, seed), 'rb') as file:
            model = pkl.load(file)
        c_index, pred_DCSM, pred_time_DCSM, rae_nc_DCSM, rae_c_DCSM = test_DCSM(model, X_test, y_test)

    print('DCSM_c_index on all test data: {:.4f}'.format(c_index))
    result_DCSM.append(c_index)
    print('DCSM_rae_nc on test data: {:.4f}'.format(rae_nc_DCSM))
    rae_nc_DCSM_list.append(rae_nc_DCSM)
    print('DCSM_rae_c on test data: {:.4f}'.format(rae_nc_DCSM))
    rae_c_DCSM_list.append(rae_c_DCSM)

    t_sample = sample_weibull(scales=pred_DCSM.squeeze(), shape=args.weibull_shape)
    cal_DCSM = calibration(predicted_samples=t_sample, t=t_test, d=e_test)
    print('DCSM_cal on test data: {:.4f}'.format(cal_DCSM))
    cal_DCSM_list.append(cal_DCSM)

    is_cluster = args.is_cluster
    if is_cluster:
        print('----------------------cluster data with DCSM------------------------')
        cluster_tags_DCSM, shape, scale = model.predict_phenotype(np.float64(X_test))
        shape = shape.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()
        X_test_list = []
        y_test_list = []
        for i in range(model.k):  # go through all the classes
            idx_i = np.where(cluster_tags_DCSM == i)[0]
            print('num in cluster {} is {}'.format(i, len(idx_i)))
            X_test_list.append(X_test[idx_i])
            y_test_list.append(y_test[idx_i])
        pval, logrank = plot_KM(y_test_list, 'DCSM', data_name, is_train=False,
                seed=seed, num_inst=args.num_inst, num_feat=args.num_feat,
                is_expert=False, shape=shape, scale=scale, t_horizon=t_test.max())
        logrank_DCSM.append(logrank)
        plot_Weibull_cdf(t_test.max(), shape, scale, data_name=data_name, seed=seed)

low_DCSM, high_DCSM = st.t.interval(0.95, df=len(result_DCSM)-1, loc=np.mean(result_DCSM), scale=st.sem(result_DCSM))
print('-----------------C Index results-----------------')
print('DCSM:{:.4f}±{:.4f} from {:.4f} to {:.4f}'.format(np.mean(result_DCSM), np.std(result_DCSM), low_DCSM, high_DCSM))
low_DCSM, high_DCSM = st.t.interval(0.95, df=len(logrank_DCSM)-1, loc=np.mean(logrank_DCSM), scale=st.sem(logrank_DCSM))
print('---------------logrank results-----------------')
print('DCSM:{:.4f}±{:.4f} from {:.4f} to {:.4f}'.format(np.mean(logrank_DCSM), np.std(logrank_DCSM), low_DCSM, high_DCSM))
print('---------------rae_nc results-----------------')
print('DCSM:{:.4f}±{:.4f}'.format(np.mean(rae_nc_DCSM_list), np.std(rae_nc_DCSM_list)))
print('---------------rae_c results-----------------')
print('DCSM:{:.4f}±{:.4f}'.format(np.mean(rae_c_DCSM_list), np.std(rae_c_DCSM_list)))
print('---------------cal results---------------------')
print('DCSM:{:.4f}±{:.4f}'.format(np.mean(cal_DCSM_list), np.std(cal_DCSM_list)))

e = int(time.perf_counter() - start_time)
print('Elapsed Time: {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
