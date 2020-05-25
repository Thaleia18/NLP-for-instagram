import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
from ipywidgets import widgets, interact, interactive, fixed, interact_manual

def plotclusterscorr(df,col_of_int,clustername):
    """
    This function plots correlation between clustes and a given column using matplot. 
  
    Parameters: 
    df : pandas data frame
    col_of_int (str) :string with the name of the column to correlate with clusters
    clustername (str): name of cluster to correlate
    Returns: plot using matplotlib
    """  
    cls =df.groupby([col_of_int,clustername]).size()
    clusters = cls.groupby(level=0).apply(lambda x: x / float(x.sum()))

    fig2, ax2 = plt.subplots(figsize = (10,8))
    sns.heatmap(clusters.unstack(level = col_of_int), ax = ax2, cmap = 'Blues')

    ax2.set_xlabel(col_of_int, fontdict = {'weight': 'bold', 'size': 2})
    ax2.set_ylabel(clustername, fontdict = {'weight': 'bold', 'size': 24})
    for label in ax2.get_xticklabels():
        label.set_size(12)
        label.set_weight("bold")
    for label in ax2.get_yticklabels():
        label.set_size(16)
        label.set_weight("bold")
        
        
def plotjoint(df, clusters, color_code, x_col, y_col, xlim, ylim):
    dict_clusters=color_code
    grid = sns.JointGrid(x=x_col, y=y_col, data=df)

    g = grid.plot_joint(sns.scatterplot, hue=clusters, data=df, palette=dict_clusters)
    plt.ylim(1, ylim)
    plt.xlim(1, xlim)
    for key, value in dict_clusters.items():
        sns.kdeplot(df.loc[df[clusters]==key, x_col], ax=g.ax_marg_x, legend=False, color=value)
        sns.kdeplot(df.loc[df[clusters]==key, y_col], ax=g.ax_marg_y, vertical=True, legend=False, color=value)