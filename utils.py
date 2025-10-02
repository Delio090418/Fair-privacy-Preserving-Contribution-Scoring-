import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
import scipy.stats as sc
import random
from loggers import logger_f#metricslogger
from itertools import combinations
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib





def transform_dict_to_lists(dic,clients):
        """This function take a dic that contains the evluation of each coalion
        And output this to the format of binary tuples and acc"""

        sets = []
        accuracies = []

        for key, value in dic.items():
            binary_list = [0] * clients
            for index in key:
                binary_list[index] = 1
            sets.append(binary_list)
            accuracies.append(value)

        return sets, accuracies


def combine_plot(groups=list, categories=list, path=str,title=str, name=str):
        """
        groups: list of lists with c.sco (e.g. SV, I1I,..)
        categories: list of name (e.g. SV, I1I,..)
        path: where to save the image
        title: of the plot
        name: of the file.png
        output a plot with x axis the c.score and y the values for each client
        """
        # Data setup
        matrix = np.array(groups)

        # Number of categories and width for each bar
        x = np.arange(len(categories))
        width = 0.8 / matrix.shape[1]  # Adjusts spacing between bars based on the number of groups

        # Create figure
        plt.figure(figsize=(8, 5))

        # Define a color map
        base_colors = plt.get_cmap('tab10').colors
        colors = ListedColormap(base_colors[:matrix.shape[1]])

        # Plot each group's bars dynamically
        bars_list = []
        for i in range(matrix.shape[1]):
            bars = plt.bar(x + (i - matrix.shape[1] / 2) * width, matrix[:, i], width, label=f'Hospital {chr(65 + i)}', color=colors(i))
            bars_list.append(bars)

        # Labels, legend, and tick marks
        plt.xlabel('Metric')
        plt.ylabel('Scores')
        plt.title(f"{title}")
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a light grid for readability

        plt.savefig(f'{path}/{name}')
        plt.close()

def shapley(clients, groups, acc, path,retraining=None):
        '''
        compute the Shapley Value
            (input)  clients: client number
            (input)  groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
            (input)  acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
            (output) score:   Shapley value of the players
        '''
        scores = np.zeros(clients)
        for i in range(clients):
            tmp = 0
            for j, subset in enumerate(groups):
                if subset[i] == 0:
                    continue
                subset_without_i = np.copy(subset)
                subset_without_i[i] = 0
                subset_index = j
                idx = [k for k, l in enumerate(groups) if all(x == y for x, y in zip(l, subset_without_i))]
                subset_without_i_index = idx[0]
                marginal_contribution = acc[subset_index] - acc[subset_without_i_index]
                subset_size = np.sum(subset) - 1
                weight = (math.factorial(subset_size) * math.factorial(clients - subset_size - 1)) / math.factorial(clients)
                tmp += weight * marginal_contribution
            scores[i] = tmp
        if retraining==None:
            logger_f(f"SV: {scores}",f"{path}/values/shapleys.log")
        else:
            logger_f(f"SV_retraining: {scores}",f"{path}/values/shapleys.log")
        #svlogger.info(f"SV: {scores}")
        return scores.tolist()



def pri_ce(clients, groups, acc, path):
        '''
        compute the privacy-preserving contribution scores
            (input)  clients: client number
            (input)  groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
            (input)  acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
            (output) i1i:     include-one-in
            (output) l1o:     leave-one-out
            (output) ieei:    include-everybody-else-in
            (output) leeo:    leave-everybody-else-out
        '''
        i1i = np.zeros(clients)
        l1o = np.zeros(clients)
        f_l1o= np.zeros(clients)
        f_i1i= np.zeros(clients)
        ieei = np.zeros(clients)
        leeo = np.zeros(clients)
        include = np.zeros(clients)
        leave = np.zeros(clients)
        for i, subset in enumerate(groups):
            if np.sum(subset) == 0:
                null = acc[i]
            if np.sum(subset) == clients:
                grand = acc[i]
            if np.sum(subset) == 1:
                tmp = [j for j, k in enumerate(subset) if k == 1]
                include[tmp[0]] = acc[i]
            if np.sum(subset) == clients - 1:
                tmp = [j for j, k in enumerate(subset) if k == 0]
                leave[tmp[0]] = acc[i]
        for i in range(clients):
            i1i[i] = include[i] - null
            l1o[i] = grand - leave[i]
            for j in range(clients):
                ieei[i] += (leave[j] - null)
                leeo[i] += (grand - include[j])
            ieei[i] -= (leave[i] - null)
            leeo[i] -= (grand - include[i])
        ieei= ieei/(clients - 1) ** 2
        leeo= leeo/(clients - 1) ** 2
        #beta_ieei=np.sum(ieei)
        #beta_leeo=np.sum(leeo)
        #ieei= ieei*(grand/beta_ieei)
        #leeo= leeo*(grand/beta_leeo)
        se = (i1i + l1o)/2
        ee = (ieei + leeo)/2
        ppce = (se + ee)/2
        beta_l1o=np.sum(l1o)
        beta_i1i=np.sum(i1i)
        for i in range(clients):
            f_l1o[i] = (l1o[i]*grand)/ beta_l1o
            f_i1i[i] = (i1i[i]*grand)/ beta_i1i
        logger_f(f"f_i1i: {f_i1i.tolist()}",f"{path}/values/ppce_global.log")
        logger_f(f"f_l1o: {f_l1o.tolist()}",f"{path}/values/ppce_global.log")
        logger_f(f"i1i: {i1i.tolist()}",f"{path}/values/ppce_global.log")
        logger_f(f"l10: {l1o.tolist()}",f"{path}/values/ppce_global.log")
        logger_f(f"ie2i: {ieei.tolist()}",f"{path}/values/ppce_global.log")
        logger_f(f"le2o: {leeo.tolist()}",f"{path}/values/ppce_global.log")
        logger_f(f"se: {se.tolist()}",f"{path}/values/ppce_global.log")
        logger_f(f"ee: {ee.tolist()}",f"{path}/values/ppce_global.log")
        logger_f(f"PPce: {ppce.tolist()}",f"{path}/values/ppce_global.log")
        return [i1i.tolist(), l1o.tolist(), ieei.tolist(), leeo.tolist(),se.tolist(),ee.tolist(),ppce.tolist(),f_i1i.tolist(), f_l1o.tolist()]


# def pri_ce(clients, groups, acc, path):
#         '''
#         compute the privacy-preserving contribution scores
#             (input)  clients: client number
#             (input)  groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
#             (input)  acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
#             (output) i1i:     include-one-in
#             (output) l1o:     leave-one-out
#             (output) ieei:    include-everybody-else-in
#             (output) leeo:    leave-everybody-else-out
#         '''
#         i1i = np.zeros(clients)
#         l1o = np.zeros(clients)
#         ieei = np.zeros(clients)
#         leeo = np.zeros(clients)
#         include = np.zeros(clients)
#         leave = np.zeros(clients)
#         for i, subset in enumerate(groups):
#             if np.sum(subset) == 0:
#                 null = acc[i]
#             if np.sum(subset) == clients:
#                 grand = acc[i]
#             if np.sum(subset) == 1:
#                 tmp = [j for j, k in enumerate(subset) if k == 1]
#                 include[tmp[0]] = acc[i]
#             if np.sum(subset) == clients - 1:
#                 tmp = [j for j, k in enumerate(subset) if k == 0]
#                 leave[tmp[0]] = acc[i]
#         for i in range(clients):
#             i1i[i] = include[i] - null
#             l1o[i] = grand - leave[i]
#             for j in range(clients):
#                 ieei[i] += (leave[j] - null)
#                 leeo[i] += (grand - include[j])
#             ieei[i] -= (leave[i] - null)
#             leeo[i] -= (grand - include[i])
#         ieei= ieei / (clients - 1) ** 2
#         leeo= leeo / (clients - 1) ** 2
#         se = (i1i + l1o)/2
#         ee = (ieei + leeo)/2
#         ppce = (se + ee)/2
#         logger_f(f"i1i: {i1i.tolist()}",f"{path}/values/ppce_global.log")
#         logger_f(f"l10: {l1o.tolist()}",f"{path}/values/ppce_global.log")
#         logger_f(f"ie2i: {ieei.tolist()}",f"{path}/values/ppce_global.log")
#         logger_f(f"le2o: {leeo.tolist()}",f"{path}/values/ppce_global.log")
#         logger_f(f"se: {se.tolist()}",f"{path}/values/ppce_global.log")
#         logger_f(f"ee: {ee.tolist()}",f"{path}/values/ppce_global.log")
#         logger_f(f"PPce: {ppce.tolist()}",f"{path}/values/ppce_global.log")
#         return [i1i.tolist(), l1o.tolist(), ieei.tolist(), leeo.tolist(),se.tolist(),ee.tolist(),ppce.tolist()]
        #return i1i, l1o, ieei, leeo, se, ee, ppce    




def effect(clients, groups, acc):
        '''
        compute the privacy-preserving contribution scores
            (input)  clients: client number
            (input)  groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
            (input)  acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
            (output) i1i:     include-one-in
            (output) l1o:     leave-one-out
            (output) ieei:    include-everybody-else-in
            (output) leeo:    leave-everybody-else-out
        '''
        efieei = np.zeros((clients,clients))
        efleeo = np.zeros((clients,clients))
        efee = np.zeros((clients,clients))
        ieei = np.zeros(clients)
        leeo = np.zeros(clients)
        include = np.zeros(clients)
        leave = np.zeros(clients)
        for i, subset in enumerate(groups):
            if np.sum(subset) == 0:
                null = acc[i]
            if np.sum(subset) == clients:
                grand = acc[i]
            if np.sum(subset) == 1:
                tmp = [j for j, k in enumerate(subset) if k == 1]
                include[tmp[0]] = acc[i]
            if np.sum(subset) == clients - 1:
                tmp = [j for j, k in enumerate(subset) if k == 0]
                leave[tmp[0]] = acc[i]
        for i in range(clients):
            for j in range(clients):
                ieei[i] += leave[j] - null
                leeo[i] += grand - include[j]
            ieei[i] -= leave[i] - null
            leeo[i] -= grand - include[i]
        ieei= ieei / (clients - 1) ** 2
        leeo= leeo / (clients - 1) ** 2
        ee = (ieei + leeo)/2
        for i in range(clients):
            for j in range(clients):
                if i==j:
                    efieei[i,j]=0
                    efleeo[i,j]=0
                    efee[i,j]=0
                else:
                    efieei[i,j]=(leave[i] - null)/ieei[j]
                    efleeo[i,j]=(grand - include[i])/leeo[j]
                    efee[i,j]=((grand - include[i])+(leave[i] - null))/ee[j]
        logger_f(f"effect ieei: {efieei}","effects")
        logger_f(f"effect leeo: {efleeo}","effects")
        logger_f(f"effect ee: {efee}","effects")
        return [efieei,efleeo,efee]


def affine_trans_list(lst):
    numbers=np.array(lst)
    if np.min(numbers)<0:
        tmp=numbers - np.min(numbers)
        tmp_mean=np.mean(tmp)
        trans=tmp/tmp_mean
    elif np.min(numbers)==0 and np.max(numbers)==0:
        trans=lst
    else:
        pos_mean=np.mean(lst)
        trans=lst/pos_mean
    return trans



# def ratio_error(list1,list2):
#     """input: two list of the same dimension
#     output: the ratio error of between the two lists"""
#     errors=[]
#     for i in range(len(list1)):
#         for j in range(i,len(list2)):
#             errors.append((list1[i]/(list1[j] + 1e-10)-list2[i]/(list2[j]+1e-10))**2)
#     return np.mean(errors)

            
######box plot iterations

def affine_trans(ppce):
    tmp=[]
    for list_ in ppce:
        tmp.append(affine_trans_list(list_))
    return tmp

def least_squares_error(y_true, y_pred):
    return sum((yi - y_hat) ** 2 for yi, y_hat in zip(y_true, y_pred))


def approx_quantities(sv, ppce, path):
    """
    ratio mse: the mean square error between the ratios
    this computes the correlation coefic. (ratio mse, spearman) of sv with respect to the others ppce
    """
    np_sv = np.array(sv, dtype=np.float64)
    valores = []
    sv_norm = affine_trans_list(sv)
    ppce_nor = affine_trans(ppce)
    for value in ppce:
        rmse = least_squares_error(sv_norm, ppce_nor[ppce.index(value)])
        np_value = np.array(value, dtype=np.float64)
        # Mask to filter out NaN/inf in both arrays
        mask = np.isfinite(np_sv) & np.isfinite(np_value)
        sv_clean = np_sv[mask]
        value_clean = np_value[mask]
        if len(sv_clean) < 2 or np.all(sv_clean == sv_clean[0]) or np.all(value_clean == value_clean[0]):
            spearman, pearson, kendall = 0, 0, 0
        else:
            spearman = sc.spearmanr(sv_clean, value_clean)[0]
            pearson = sc.pearsonr(sv_clean, value_clean)[0]
            kendall = sc.kendalltau(sv_clean, value_clean)[0]
        logger_f(f"nlse: {rmse}, spearman: {spearman}, pearson: {pearson}, kendall:{kendall}", path)
        valores.append([rmse, spearman, pearson, kendall])
    return valores              

# def approx_quantities(sv,ppce,path):
#         """
#         ratio mse:the mean square error between the ratios
#         this compute the correlation coefic. (ratio mse, spearman) of sv with respect to the others ppce
#         """
#         # path = pathlib.Path(path)
#         np_sv=np.array(sv)
#         valores=[]
#         sv_norm=affine_trans_list(sv)
#         ppce_nor=affine_trans(ppce)
#         for value in ppce:
#             rmse=least_squares_error(sv_norm, ppce_nor[ppce.index(value)])
#             np_value=np.array(value)
#             if np.all(np_sv == np_sv[0]) or np.all(np_value == np_value[0]):
#                 spearman,pearson,kendall=0,0,0
#             else:
#                 spearman= sc.spearmanr(sv, value)[0]
#                 pearson=sc.pearsonr(sv, value)[0]
#                 kendall=sc.kendalltau(sv, value)[0]
#             logger_f(f"nlse: {rmse}, spearman: {spearman}, pearson: {pearson}, kendall:{kendall}",path)
#             # metricslogger.info(f"nlse: {rmse}, spearman: {spearman}, pearson: {pearson}, kendall:{kendall}")
#             valores.append([rmse,spearman,pearson,kendall])
#         return valores

        
        


def plot_coeficits(groups=list,categories=list,path=str,title=str,name=str):
        """
        groups: list of lists with correlation coef. (e.g. ratio mse, spearman)
        categories: list of name (e.g. SV, I1I,..)
        path: where to save the image
        title: of the plot
        name: of the file.png
        output a plot with x axis the c.score and y the values of the correlations coef. for each client
        """
      
        matrix=np.array(groups)


        # Number of categories and width for each bar
        x = np.arange(len(categories))
        width = 0.2  # Adjusts spacing between bars

        # Create figure
        plt.figure(figsize=(8, 5))

         # Plot each group's bars
        bars_a = plt.bar(x - 1.5 * width, matrix[:, 0], width, label='MSE')
        bars_b = plt.bar(x - 0.5 * width, matrix[:, 1], width, label='Spearman')

        
        # Function to add labels above bars
        def add_labels(bars, shift_x=0):
            for bar in bars:
                height = bar.get_height()
                x_pos = bar.get_x() + bar.get_width() / 2 + shift_x  # Shift text
                plt.text(x_pos, height, f'{height:.2e}', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold', rotation=20)  # Rotate slightly for better readability

        # Add labels with slight horizontal shifts
        add_labels(bars_a, shift_x=-0.06)  # Move left
        add_labels(bars_b, shift_x=0.06)
            

        # Labels, legend, and tick marks
        plt.xlabel('Metric')
        plt.ylabel('Correlation')
        plt.title(f"{title}")
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a light grid for readability

        plt.savefig(f'{path}/{name}')
        plt.close()




def plot_boxplot_with_stats(data, sco_names, path,title,name):
    """
    Plots a boxplot for a list of lists, with mean and std overlayed.

    Parameters:
    - data (list of lists): A list of lists where each inner list contains data points for the boxplot.
    - sco_names (list of str): A list of names for each inner list, used as x-axis labels.
    -path: where to be saved
    -title: of the plot
    -name: of the file
    """
    # Prepare data for plotting
    means = [np.mean(lst) for lst in data]
    stds = [np.std(lst) for lst in data]

    max_range = max(means)
    min_range = min(means)
    range_ = max_range - min_range
    th = 0.2 * range_

    
    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, notch=False, patch_artist=True)

    
    
    # Overlay the mean and std
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + 4 * th, f'Mean: {mean:.2e}', horizontalalignment='center', color='black')
        plt.text(i, mean+th, f'STD: {std:.2e}', horizontalalignment='center', color='black')
    
    # Set x-tick labels
    plt.xticks(ticks=np.arange(len(sco_names)), labels=sco_names)
    
    # Set plot labels and title
    plt.xlabel('Scores')
    plt.ylabel('Values')
    plt.title(f'{title}')
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{path}/{name}')
    plt.close()


def cosine_similarity_models(model_client,model_server):
    """
    Compute the cosine similarity between the parameters of two PyTorch models.
    """
    params_a = torch.cat([p.view(-1) for p in model_client.parameters()])
    params_b = torch.cat([p.view(-1) for p in model_server.parameters()])
    
    similarity = F.cosine_similarity(params_a.unsqueeze(0), params_b.unsqueeze(0))
    
    return similarity.item()


def mean_columns_list(array):
    """
    Computes the mean across the first dimension (axis=0) of a NumPy array
    and returns a list where each entry is a column from the mean array.

    Args:
        array (np.ndarray): Input NumPy array of any dimension.

    Returns:
        list: A list of 1D NumPy arrays, each representing a column from the mean result.
    """
    # Compute the mean along axis 0
    mean_array = np.mean(array, axis=0)

    # Convert the mean array's columns into a list of 1D arrays
    columns = [mean_array[:, i] for i in range(mean_array.shape[1])]

    return columns







