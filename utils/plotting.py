import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.ticker import MaxNLocator, StrMethodFormatter, FixedLocator
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
from lifelines.statistics import multivariate_logrank_test
from lifelines import KaplanMeierFitter
import os


def _figure_path(filename: str) -> str:
    figure_dir = os.environ.get("ADACSM_FIGURE_DIR", "./Figures")
    os.makedirs(figure_dir, exist_ok=True)
    return os.path.join(figure_dir, filename)


def _maybe_show():
    if os.environ.get("ADACSM_NO_SHOW", "1") != "1":
        plt.show()


def plot_Weibull_cdf(t_horizon, shape, scale, data_name='sim', num_inst=1000, num_feat=200, seed=42):
    step = 100
    for i in range(len(shape)):
        k = shape[i]
        b = scale[i]
        s = np.zeros(step)
        t_space = np.linspace(0, t_horizon, step)
        for j in range(step):
            s[j] = np.exp(-(np.power(np.exp(b) * t_space[j], np.exp(k))))
        plt.plot(t_space, s, label='Expert Distribution {}'.format(i))
    plt.legend()
    # plt.title('Weibull CDF, Data: {}, Seed: {}'.format(data_name, seed))
    plt.title('Weibull CDF, Data: {}'.format(data_name), fontsize=16)
    
    if data_name == 'sim':
        plt.savefig(_figure_path('Weibull_cdf_#clusters{}_{}_{}x{}_seed{}.png'.
                    format(len(shape), data_name, num_inst, num_feat, seed)))
    else:
        plt.savefig(_figure_path('Weibull_cdf_#clusters{}_{}_seed{}.png'.
                    format(len(shape), data_name, seed)))
    _maybe_show()
    plt.close()


def plot_loss_c_index(results_all, lr, epoch, bs):
    # plot the loss and the C Index
    fig, ax = plt.subplots()
    ax.plot(results_all[:, 0], color='tab:red', label='train loss')
    ax.plot(results_all[:, 1], color='tab:blue', label='test loss')
    ax.set_xlabel("epoch", fontsize=14)
    ax.set_ylabel("loss", fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(results_all[:, 2], color='tab:green', label='C Index Test')
    ax2.plot(results_all[:, 3], color='tab:orange', label='C Index Train')
    ax2.set_ylabel("C Index", fontsize=14)
    ax2.plot(np.nan, color='tab:red', label='train loss')  # print an empty line to represent loss
    ax2.plot(np.nan, color='tab:blue', label='test loss')

    ax2.legend(loc=0)
    ax.grid()
    plt.title('lr: {:.2e}, epoch: {}, batch_size: {}'.format(lr, epoch, bs))
    _maybe_show()
    plt.close()


def visualize(X_train_list, X_test_list, data_name, is_normalize=0, is_TSNE=1):
    """This function is to visualize the scatter plot with clustering information"""

    X_train = np.concatenate(X_train_list)
    X_test = np.concatenate(X_test_list)
    X = np.concatenate((X_train, X_test), axis=0)
    # normalize
    if is_normalize == 1:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    if is_TSNE == 1:
        # embed using TSNE
        embeddings = TSNE(random_state=42).fit_transform(X)
    else:
        # embed using UMAP
        trans = umap.UMAP(random_state=42).fit(X)
        embeddings = trans.embedding_
        # embeddings = []


    xlim = [-100, 95]
    ylim = [-90, 90]

    # show each cluster separately on all train data
    len_train = 0
    for idx, f in enumerate(X_train_list):
        plt.scatter(embeddings[len_train:(len_train + len(f)), 0],
                    embeddings[len_train:(len_train + len(f)), 1],
                    s=5, label='Train Cluster {}'.format(idx))

        len_train += len(f)
        plt.xlim(xlim)
        plt.ylim(ylim)
    plt.title(data_name)
    # plt.legend()
    plt.xlabel('Train Data with #Clusters {}'.format(len(X_train_list)))
    _maybe_show()
    plt.close()

    # show each cluster separately on all test data
    len_test = len(X_train)
    for idx, f in enumerate(X_test_list):
        plt.scatter(embeddings[len_test:(len_test + len(f)), 0],
                    embeddings[len_test:(len_test + len(f)), 1],
                    s=5, label='Test Cluster {}'.format(idx))
        len_test += len(f)
        plt.xlim(xlim)
        plt.ylim(ylim)
    plt.title(data_name)
    # plt.legend()
    plt.xlabel('Train Data with #Clusters {}'.format(len(X_test_list)))
    _maybe_show()
    plt.close()


def plot_KM(y_list, cluster_method, data_name,
            is_train=True, is_lifelines=True,
            seed=42, num_inst=1000, num_feat=200,
            is_expert=False, shape=[], scale=[], t_horizon=10):
    """This function is to plot the Kaplan-Meier curve regarding different clusters"""

    if is_train:
        stage = 'train'
    else:
        stage = 'test'

    group_indicator = []
    for idx, cluster in enumerate(y_list):
        group_indicator.append([idx] * len(cluster))
    group_indicator = np.concatenate(group_indicator)

    if is_lifelines:
        results = multivariate_logrank_test([item[1] for item in np.concatenate(y_list)], # item 1 is the survival time
                                            group_indicator,
                                            [int(item[0]) for item in np.concatenate(y_list)]) # item 0 is the event
        chisq, pval = results.test_statistic, results.p_value
    else:
        chisq, pval = compare_survival(np.concatenate(y_list), group_indicator)

    print('Test statistic of {}: {:.4e}'.format(stage, chisq))
    print('P value of {}: {:.4e}'.format(stage, pval))
    fig = figure(figsize=(7.0, 5.0), dpi=300)
    ax = plt.gca()
    # Paper-style colors used in plot_km.py:
    # low-risk: periwinkle blue, high-risk: rose pink.
    low_risk_color = '#6C7BFF'
    high_risk_color = '#E75480'
    cluster_colors = [None] * len(y_list)
    if len(y_list) == 2:
        event_rates = []
        for cluster in y_list:
            n_events = sum(int(item[0]) for item in cluster)
            event_rates.append((n_events / len(cluster)) if len(cluster) else 0.0)
        high_risk_idx = 0 if event_rates[0] > event_rates[1] else 1
        cluster_colors[high_risk_idx] = high_risk_color
        cluster_colors[1 - high_risk_idx] = low_risk_color
    else:
        palette = plt.cm.Set2(np.linspace(0, 1, len(y_list)))
        for i in range(len(y_list)):
            cluster_colors[i] = palette[i]

    for idx, cluster in enumerate(y_list):  # each element in the y_list is a cluster
        # use lifelines' KM tool to estimate and plot KM
        # this will provide confidence interval
        if len(cluster) == 0:
            continue
        if is_lifelines:
            kmf = KaplanMeierFitter()
            if len(y_list) == 2:
                risk_name = "High risk" if cluster_colors[idx] == high_risk_color else "Low risk"
                label = f"{risk_name} (n={len(cluster)})"
            else:
                label = 'Cluster {}, #{}'.format(idx, len(cluster))
            kmf.fit([item[1] for item in cluster], event_observed=[item[0] for item in cluster], label=label)
            sf = kmf.survival_function_
            times = sf.index.to_numpy(dtype=float)
            surv = sf.iloc[:, 0].to_numpy(dtype=float)
            ax.step(
                times,
                surv,
                where='post',
                label=label,
                color=cluster_colors[idx],
                linewidth=2.8,
                alpha=0.9,
            )
            et = kmf.event_table
            censor_mask = et['censored'] > 0
            if censor_mask.any():
                censor_times = et.index[censor_mask].to_numpy(dtype=float)
                censor_surv = kmf.predict(censor_times).to_numpy(dtype=float)
                ax.scatter(
                    censor_times,
                    censor_surv,
                    marker='|',
                    s=45,
                    color=cluster_colors[idx],
                    alpha=0.95,
                )
        else:
            # use scikit-survival's KM tool to estimate and plot KM
            # this does not provide confidence interval
            x, y = kaplan_meier_estimator([item[0] for item in cluster], [item[1] for item in cluster])
            ax.step(
                x,
                y,
                where="post",
                label='Cluster {}, #{}'.format(idx, len(cluster)),
                color=cluster_colors[idx],
                linewidth=2.8,
                alpha=0.9,
            )

    if is_expert:
        step = 100
        for i in range(len(shape)):
            k = shape[i]
            b = scale[i]
            s = np.zeros(step)
            t_space = np.linspace(0, t_horizon, step)
            for j in range(step):
                s[j] = -(np.power(np.exp(b) * t_space[j], np.exp(k)))
            plt.plot(t_space, s, label='Expert Distribution {}'.format(i))

    plt.title(r"Log-Rank $\chi^2$ = {:.2f}".format(chisq), fontsize=25, pad=16)
    plt.xlabel("Time (months)", fontsize=22, labelpad=10)
    plt.ylabel("Survival Probability", fontsize=22, labelpad=12)
    legend = plt.legend(fontsize=21, loc='lower left', framealpha=0.95, fancybox=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#D9D9D9')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#6A5E78')
    ax.spines['bottom'].set_color('#6A5E78')
    ax.tick_params(axis='both', which='major', labelsize=21, length=6.5, width=1.3, color='#6A5E78')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=4))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    y_ticks = np.linspace(0.0, 1.0, 6)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if data_name == 'sim':
        plt.savefig(_figure_path('{}_{}_KM_plot_#clusters{}_{}_{}x{}_seed{}.png'.
                    format(cluster_method, stage, len(y_list), data_name, num_inst, num_feat, seed)))
    else:
        plt.savefig(_figure_path('{}_{}_KM_plot_#clusters{}_{}_seed{}.png'.
                    format(cluster_method, stage, len(y_list), data_name, seed)))
    _maybe_show()
    plt.close()
    return pval, chisq


