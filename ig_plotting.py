import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
from ipywidgets import widgets, interact, interactive, fixed, interact_manual
from sklearn.cluster import KMeans

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

def dfcorr(df,col_of_int,clustername):
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
    return
    
def plotjoint(df, clusters, color_code, x_col, y_col, xlim, ylim, atitle):
     # plt.title(atitle, fontsize=40)
    dict_clusters=color_code
    grid = sns.JointGrid(x=x_col, y=y_col, data=df)

    g = grid.plot_joint(sns.scatterplot, hue=clusters, data=df, palette=dict_clusters)
    plt.ylim(1, ylim)
    plt.xlim(1, xlim)
    for key, value in dict_clusters.items():
        sns.kdeplot(df.loc[df[clusters]==key, x_col], ax=g.ax_marg_x, legend=False, color=value)
        sns.kdeplot(df.loc[df[clusters]==key, y_col], ax=g.ax_marg_y, vertical=True, legend=False, color=value)
        
        
def finding_k(vector,mink,maxk,step,atitle):
    """ 
    This function uses the elbow method to find the optimum number of clusters. 
  
    It calculates the Inertia (Sum of distances of samples to their closest cluster center) and plots it as
    function of k. 
  
    Parameters: 
    vector : Sample
    mink : min k value 
    maxk : max k value
    step : to go from kmin to kmax
    atitle : title for the plot
  
    Returns: elbow plot using matplotlib
  
    """
    sse = {}
    for k in range(mink, maxk, step):
        kmeans = KMeans(n_clusters=k, n_init = 10, max_iter=100,n_jobs = -1).fit(vector)
        #data["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_ # Inertiar
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of clusters")
    
def clusterwords(vector, vectorizing,clustermodel):
    X = pd.DataFrame(vector.toarray(), columns=vectorizing.get_feature_names())  # columns argument is optional
    X['Cluster'] = clustermodel.labels_  # Add column corresponding to cluster number
    word_frequencies_by_cluster = X.groupby('Cluster').sum()

    y = pd.DataFrame(word_frequencies_by_cluster.columns.values[np.argsort(-word_frequencies_by_cluster.values, axis=1)[:, :10]], 
                      index=word_frequencies_by_cluster.index,
      columns = ['1st','2nd','3rd','4th', '5th', '6th','7th','8th','9th','10th']).reset_index()
    return y

designerbrands = ['viviennewestwood', 'dvf', 'alexanderwang', 'sandro', 'katespade', 'paulsmith', 'rebeccaminkoff', 'theory', 'marcbymarcjacobs', 'maisonmargiela', 'iro', 'marcjacobs', 'vince', 'isabelmarant', 'stellamccartney', 'alexandermcqueen', 'coach', 'michaelkors', 'acnestudios']
highstreetbrand = ['forever21', 'mango', 'topshop', 'hollister', 'jcrew', 'abercrombie', 'americaneagle', 'zara', 'calvinklein', 'uniqlo', 'gap', 'urbanoutfitters', 'americanapparel']
megabrands = ['louisvuitton', 'tiffany', 'chanel', 'cartier', 'gucci', 'prada', 'hermes', 'burberry']
smallbrands = ['cesareattolini', 'brioni', 'brunellocucinelli', 'fabianafilippi', 'kiton', 'ermenegildozegna', 'loropiana', 'nancygonzalez']

def color_word(word, *args, **kwargs):
        if (word in designerbrands):
            color = '#000080' 
        elif (word in highstreetbrand):
            color = '#800080'
        elif (word in megabrands):
            color = '#FF6347'
        else:
            color = '#008080' 
        return color
    
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def cloudgenerator(clusternumber,df, clustercolumn):
    for i in range(0,clusternumber):
        wcloud=WordCloud(max_font_size=40, max_words=20, background_color="white",collocations=False,color_func=color_word).generate(' '.join(df[df[clustercolumn] == i]['brandname']))
        wcloud.generate_from_frequencies
        plt.figure()
        plt.title('Cluster %d' %i, fontsize=40)
        plt.imshow(wcloud, interpolation="bilinear")
        plt.axis("off")
        
from textblob import TextBlob

def analysis(df, text_column, polarity_column1, subjectivity_column2):
    newdf = pd.DataFrame()
    newdf[polarity_column1] = df[text_column].apply(lambda text: TextBlob(text).polarity)
    newdf[subjectivity_column2] = df[text_column].apply(lambda text: TextBlob(text).subjectivity)
    return newdf