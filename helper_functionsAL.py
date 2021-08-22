
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

import ptype.tests.utils as ut

def quick_timestamp():
    now = datetime.datetime.now()
    print ("Cell Run At: ",now.strftime("%Y-%m-%d %H:%M:%S"))
    return now


def test_function():
    for i in range(0,3):
        print('Hi! my name is,')
    print('Chika chika slim shady.')


def convert_to_date(annots,option='string'):
    if option == 'string':
        if 'date' in annots:
            return 'date'
        else:
            return annots
    elif option == 'list':
        returnList = ['date' if 'date' in annot else annot for annot in annots]
        return returnList
    else:
        return 'option not specified'


def get_total_confusion_values(annotations_dict,predictions_dict,ptype_types,option='dates_together'):

  # Gives total no of TP, FP, TN and TP for each type. 
    if option == 'dates_separate':
        typ_labels = ptype_types
    else:
        typ_labels = np.unique(['date' if 'date' in typ else typ for typ in ptype_types])

    # Initialise the dictionary with zero values for each permutation
    tot_conf = {t:{'TP':0,'FP':0,'TN':0,'FN':0} for t in typ_labels}

    cats = ['FN','TN','FP','TP']
    
    for annot in annotations_dict.keys(): 
        scores = ut.evaluate_model_type(annotations_dict[annot],predictions_dict[annot],option=option)
      #print(scores)
        for i,typ in enumerate(scores):
            for cat in cats:
                tot_conf[typ][cat] += scores[typ][cat]

    return tot_conf



def prep_for_confusion_mat(annotations_dict,predictions_dict,option='dates_together'):
    y = []
    y_pred = []
    labels = []

    for dataset in annotations_dict:
        if option == 'dates_together':
            convert_annots = convert_to_date(annotations_dict[dataset],option='list')
            convert_predicts = convert_to_date(predictions_dict[dataset],option='list')
        elif option == 'dates_separate':
            convert_annots = annotations_dict[dataset]
            convert_predicts = predictions_dict[dataset]
        #labels = [item for item in annotations_testC[dataset] if item not in labels]
        y = y + convert_annots
        y_pred = y_pred + convert_predicts
    return y,y_pred

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)