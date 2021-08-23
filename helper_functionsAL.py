
from sklearn.metrics import confusion_matrix
from scipy import interp
from itertools import cycle

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import os

import tests.utils as ut

def quick_timestamp():
    now = datetime.datetime.now()
    print ("Cell Run At: ",now.strftime("%Y-%m-%d %H:%M:%S"))
    return now


def test_function():
    for i in range(0,3):
        print('Hi! my name is,')
    print('Chika chika slim shady.')


def header_option(HeaderOptionIn):
    if HeaderOptionIn == 'unknown':
        return 'infer'  #  None 
    elif HeaderOptionIn == 'None':
        return None
    else:
        return HeaderOptionIn


def import_data_and_annotations(fullFilePath,rowData,option=None):
    # Where,
    # fullFilePath is a string containng the path of the folder containing data.csv and annotations.csv
    # rowData is the entry from the lookup .csv file which specifies import parameters for the dataset.     
    #Import the .csv files
    # Import should be done exactly this way to avoid errors resulting from the dataset
    tmpDF = pd.read_csv(os.path.join(fullFilePath,'data.csv'),header=header_option(rowData['Header Option']),encoding=rowData['Encoding'],dtype=str)
        
    #print('Data Columns',tmpDF.columns)
    annotsDF = pd.read_csv(os.path.join(fullFilePath,'annotations.csv'),dtype=str)
    annotsDF = annotsDF[['Column','Type']]
            
    AnnotsList = [item for item in annotsDF['Column'].values if not pd.isnull(item) ] # item != 'nan'] # not np.isnan(item)] #  'nan']
        
    # To filter out unsupported types at this stage, insert following code here. 
    # # # AnnotsList = [item for item in AnnorsList if item not in ptype.types] - where a list of types has been passed to the function
        
    annotsDF = annotsDF[annotsDF['Column'].isin(AnnotsList)]
        
    annotsDF['Column'] = annotsDF['Column'].str.replace(' ','')
              
    # Cleaning - remove spaces from data headers and ensure they are of type string 
    dataColList = tmpDF.columns
    cleanDataColList = [str(item).replace(' ','') for item in dataColList]
        
    tmpDF.columns = cleanDataColList
        
    if option == 'verbose':
        chkList = []
        for i,item in enumerate(list(tmpDF.columns)):
            try:
                #currentAnnotation = str(AnnotsList[i])
                chk = item == annotsDF['Column'][i]
                chkList.append(chk)
            except:
                print("!!!! something didn't work there !!!!!!")
                chk = False
                # Now total the errors. 
                chkNum = [err for err in chkList if err == False]
                print('========== N Errors:',len(chkNum),'===============')
        
    return tmpDF,annotsDF


def model_run(ptype,DatasetRef,option=None,filePrefix):
    # Initialise objects to be returned.
    ptype_Schemas = []
    annotations_dict = {}
    predictions_dict = {}
    datasets_list = []

    for rowNum,rowData in DatasetRef.iterrows():
        print(rowNum)
    
        if not rowData['Skip']:
        #if str(rowData['datasetID']) in(focusList):
            # Create a full path folder using the specified folder (filePrefix) and Location info from DatasetLookup
            fullFilePath = os.path.join(filePrefix,rowData['Location (relative)'])
            
            # User can specify if they want output shown.
            if option == 'verbose':    
                #print(fullFilePath)
                print(rowData['datasetID'],'--',rowData['Location (relative)'])

            dataDF,annotsDF = import_data_and_annotations(fullFilePath,rowData)
            
            if option =='verbose':
                print(annotsDF.shape)
            
            schema = ptype.schema_fit(dataDF)
            ptype_Schemas.append(schema)
            #schema = ptype_too.shema_fit(dataDF)
            #ptype_Schemas_too.append(shema)
            
            if option == 'verbose':
                print(rowData['datasetID'],'schema fit Complete!')
            
            # Save the annotations file for later reference.
            # annotations_DFs.append(annotsDF)
            annotations_dict[rowData['datasetID']] = annotsDF.Type.tolist()
            # Type inference - schema contains the ptype predictions...
            schemaRatios = schema.show_ratios()
            # ... a list of which are accessed like this.
            predictions = schemaRatios[schemaRatios.index=='type'].values[0].tolist()
            # Store the predictions in a dictionary, indexed by the datasetID
            predictions_dict[rowData['datasetID']] = predictions
            # Add the name of the dataset to a list that is returned for future reference. 
            datasets_list.append(rowData['datasetID'])
            print('-------')
            
    return datasets_list,predictions_dict,annotations_dict,ptype_Schemas



# ======= Model Evaluation Code ======= # 

# Convert to date is used to change any label or prediction containing 'date' to have the value 'date'
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


# Generates matrices for multi-class ROC-AUC
def get_evaluation_matrices(annotations_dict,predict_dict,classes,showOutput=False):
  y_true_matrix = np.zeros(shape=(0,len(classes)),dtype=int)
  y_score_matrix = np.zeros(shape=(0,len(classes)),dtype=int)
  tot=0
  for i,key in enumerate(predict_dict):
    true = annotations_dict[key]
    for j,pred in enumerate(predict_dict[key]):
      t_label = true[j]
      t_label = hf.convert_to_date(t_label)
      # print(true)
      pred = hf.convert_to_date(pred) # relabel predictions as dates for assessment. 
      if t_label in classes:
        p_i = [i for i,label in enumerate(classes) if label == pred]
        t_i = [i for i,label in enumerate(classes) if label == t_label]
        tot += 1
        #print(pred,'-',p_i)
        p_entry = np.zeros(shape=(1,len(classes)),dtype=int)
        p_entry[0,p_i] = 1
        t_entry = np.zeros(shape=(1,len(classes)),dtype=int)
        t_entry[0,t_i] = 1
        #print(tmp_row)
        y_true_matrix = np.append(y_true_matrix,t_entry,axis=0)
        y_score_matrix = np.append(y_score_matrix,p_entry,axis=0)
        #bin_class_matrix = 
      else:
        if showOutput:
          print(t_label,'not in classes list')
  print('tot count:',tot,' -- matrix dimensions',y_true_matrix.shape, y_score_matrix.shape)
  return y_true_matrix, y_score_matrix



# Happily copied from https://github.com/DTrimarchi10/confusion_matrix - what a helpful fellow.
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


# Code sourced from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def show_roc_auc(y_true_matrix,y_score_matrix,classes,title):

  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_true_matrix[:, i], y_score_matrix[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(y_true_matrix.ravel(), y_score_matrix.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(len(classes)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC
  mean_tpr /= len(classes) # n_classes

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  # Plot all ROC curves
  plt.figure()
  plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

  plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

  colors = cycle(['aqua', 'darkorange','pink' ,'purple','red'])
  for i, color in zip(range(len(classes)), colors):
    plt.plot(fpr[i], tpr[i], color=color, 
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(classes[i], roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--')#, lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(title)
  plt.legend(loc="lower right")
  plt.show()
