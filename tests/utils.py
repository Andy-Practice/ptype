from collections import Counter, OrderedDict
import glob
import numpy as np
import pandas as pd
import datetime # AL-Added to timestamp output files. 

print('Loading utils.py')

# AL-Updated functions to allow user to specify whether they want results with all date types presented
# together (option='dates_together') or individually (option='dates_separate'). The former is the default.

def evaluate_model_type(annotations, predictions,option='dates_together'):

    if option == 'dates_separate':
        types = ["boolean", "float", "integer", "string","date-iso-8601", "date-eu","date-non-std-subtype","date-non-std"]
    else: 
        types = ["boolean", "date", "float", "integer", "string"]

    type_rates = {t: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for t in types}

    if not option == 'dates_separate':
        annotations = ['date' if 'date' in val else val for val in annotations]
        predictions = ['date' if 'date' in val else val for val in predictions] 

        # predictions = [
        #     prediction.replace("date-eu", "date")
        #     .replace("date-iso-8601", "date")
        #     .replace("date-non-std-subtype", "date")
        #     .replace("date-non-std", "date")
        #     for prediction in predictions
        # ]

        # annotations = [
        #     annotation.replace("date-eu", "date")
        #     .replace("date-iso-8601", "date")
        #     .replace("date-non-std-subtype", "date")
        #     .replace("date-non-std", "date")
        #     for annotation in annotations
        # ]

    # find columns whose types are not supported by ptype
    ignored_columns = np.where(
        (np.array(annotations) != "all identical")
        & (np.array(annotations) != "gender")
        & (np.array(predictions) != "all identical")
        & (np.array(predictions) != "unknown")
    )[0]

    for t in types:
        # print(t)
        # print(ignored_columns)
        # print(annotations)
        # print(predictions)
        y_true = (np.array(annotations) == t)[ignored_columns]
        y_score = (np.array(predictions) == t)[ignored_columns]

        type_rates[t]["TP"] = sum(y_true * y_score)
        type_rates[t]["FP"] = sum(not_vector(y_true) * y_score)
        type_rates[t]["TN"] = sum(not_vector(y_true) * not_vector(y_score))
        type_rates[t]["FN"] = sum(y_true * not_vector(y_score))

    return type_rates


def evaluate_predictions(annotations, type_predictions,option='dates_together'):
    # the column type counts of the datasets
    [_, _, total_cols] = get_type_counts(type_predictions, annotations,option)

    Js, overall_accuracy = get_evaluations(annotations, type_predictions,option)
    overall_accuracy_to_print = {
        method: {"overall-accuracy": float_2dp(overall_accuracy[method] / total_cols)}
        for method in overall_accuracy
    }
    print("overall accuracy: ", overall_accuracy_to_print)
    print("Jaccard index values: ", {t: Js[t]["ptype"] for t in Js})

    df1 = pd.DataFrame.from_dict(Js, orient="index")
    df2 = pd.DataFrame.from_dict(overall_accuracy_to_print, orient="index").T
    df = df2.append(df1)
    column_type_evaluations = "tests/column_type_evaluations.csv"
    expected = pd.read_csv(column_type_evaluations, index_col=0)

    # AL-Added so that model evaluations are always exported to a unique, timestamped csv file.  
    now = datetime.datetime.now()
    outputfilename = 'column_type_evaluations-'+now.strftime("%y%m%d-%H%M%S")+'.csv'
    df.to_csv(path_or_buf='tests/'+outputfilename)

    # AL-Added so that the model results are returned as a dataframe
    return df
        
    # if not expected.equals(df):        
    #     df.to_csv(path_or_buf=column_type_evaluations + ".new")
    #     raise Exception(f"{column_type_evaluations} comparison failed.")


def float_2dp(n: float):
    """Round a float to 2 decimal places, preserving float-hood. Probably a better way to do this."""
    return np.float64("{:.2f}".format(n))


def get_evaluations(_annotations, _predictions,option='dates_together'):
    methods = ["ptype"]
    dataset_names = list(_predictions.keys())

    if option == 'dates_separate':
        types = types = ["boolean", "float", "integer", "string","date-iso-8601", "date-eu","date-non-std-subtype","date-non-std"]
    else:
        types = ["boolean", "date", "float", "integer", "string"]

    Js = {}
    overall_accuracy = {method: 0 for method in methods}
    for t in types:

        J = {}
        for method in methods:

            tp, fp, fn = 0.0, 0.0, 0.0
            for dataset_name in dataset_names:
                temp = evaluate_model_type(
                    # AL - Following ammended, as my code passes a dictionary object indexed by dataset_name
                    #_annotations[dataset_name], _predictions[dataset_name].values()
                    _annotations[dataset_name], _predictions[dataset_name],option=option#.values()

                )
                tp += temp[t]["TP"]
                fp += temp[t]["FP"]
                fn += temp[t]["FN"]

            overall_accuracy[method] += tp
            J[method] = float_2dp(tp / (tp + fp + fn))
        Js[t] = J

    return Js, overall_accuracy


def get_type_counts(predictions, annotations,option='dates_together'):
    
    if option == 'dates_separate':
        _types = ["boolean", "float", "integer", "string","date-iso-8601", "date-eu","date-non-std-subtype","date-non-std"]
    else: 
        _types = ["boolean", "date", "float", "integer", "string"]
    
    dataset_counts = OrderedDict()
    total_test = {t: 0 for t in _types}

    for dataset_name in predictions:

        true_values = annotations[dataset_name]
        # AL - Following ammended, as my code passes a dictionary object indexed by dataset_name
        #ptype_predictions = predictions[dataset_name].values()
        ptype_predictions = predictions[dataset_name] # .values()

        if not option == 'dates_separate':
            true_values = ['date' if 'date' in val else val for val in true_values]
            ptype_predictions = ['date' if 'date' in val else val for val in ptype_predictions] 

            # true_values = [
            #     true_value.replace("date-eu", "date")
            #     .replace("date-iso-8601", "date")
            #     .replace("date-non-std-subtype", "date")
            #     .replace("date-non-std", "date")
            #     for true_value in true_values
            # ]
            # ptype_predictions = [
            #    prediction.replace("date-eu", "date")
            #     .replace("date-iso-8601", "date")
            #     .replace("date-non-std-subtype", "date")
            #     .replace("date-non-std", "date")
            #     for prediction in ptype_predictions
            # ]

        ignored_columns = np.where(
            (np.array(true_values) != "all identical")
            & (np.array(true_values) != "gender")
            & (np.array(ptype_predictions) != "all identical")
            & (np.array(ptype_predictions) != "unknown")
        )[0]

        counts = Counter(np.array(true_values)[ignored_columns])
        for t in _types:
            if t not in list(counts.keys()):
                counts[t] = 0

        # Counters are unordered, so for deterministic output we sort via a list
        dataset_counts[dataset_name] = dict(sorted(counts.items()))

        total_test = {
            t: total_test[t] + dataset_counts[dataset_name][t] for t in _types
        }

    total_cols = sum(
        [total_test[t] for t in _types]
    )

    return [total_test, dataset_counts, total_cols]


# added later - needs a pass over
def not_vector(X):
    return np.array([not x for x in X])
