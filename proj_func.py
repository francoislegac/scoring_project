##FONCTIONS PROJET DATA SCIENCE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scikitplot.helpers import cumulative_gain_curve
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, f1_score
from sklearn.metrics import roc_curve, roc_auc_score


#SEVERAL LIFT CHART ON THE SAME PLOT
def plot_cumulative_gain(y_true, y_probas, title='Cumulative Gains Curve',ax=None, figsize=None, title_fontsize="large",text_fontsize="medium"):
    """Generates the Cumulative Gains Plot from labels and scores/probabilities
    The cumulative gains chart is used to determine the effectiveness of a
    binary classifier. A detailed explanation can be found at
    http://mlwiki.org/index.php/Cumulative_Gain_Chart. The implementation
    here works only for binary classification.
    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.
        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.
        title (string, optional): Title of the generated plot. Defaults to
            "Cumulative Gains Curve".
        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.
        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.
        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".
        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".
    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.
    Example:
        >>> import scikitplot as skplt
        >>> lr = LogisticRegression()
        >>> lr = lr.fit(X_train, y_train)
        >>> y_probas = lr.predict_proba(X_test)
        >>> skplt.metrics.plot_cumulative_gain(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()
        .. image:: _static/examples/plot_cumulative_gain.png
           :align: center
           :alt: Cumulative Gains Plot
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate Cumulative Gains for data with '
                         '{} category/ies'.format(len(classes)))

    # Compute Cumulative Gain Curves
    #percentages, gains1 = cumulative_gain_curve(y_true, y_probas[:, 0],
    #                                            classes[0])
    percentages, gains2 = cumulative_gain_curve(y_true, y_probas[:, 1],
                                                classes[1])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    #ax.plot(percentages, gains1, lw=3, label='Class {}'.format(classes[0]))
    ax.plot(percentages, gains2, lw=3, label='Class {}'.format(classes[1]))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.set_xlabel('Percentage of sample', fontsize=text_fontsize)
    ax.set_ylabel('Gain', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')
    #ax.legend(loc='lower right', fontsize=text_fontsize)

    return ax
def plot_several_lifts(arg_models, x_train, x_test, y_train, y_test):
    '''
    arg_models = la liste de model que tu veux tester
    '''
    fig ,ax = plt.subplots()
    legend = []
    for model in arg_models:
        x = {
            'label': 'LR %s, C = %.2f ' %(model.get_params()['penalty'], model.get_params()['C']),
            'model': model, 
        }
        legend.append(x['label'])
        x['model'].fit(x_train, y_train) # train the model
        y_pred_proba =x['model'].predict_proba(x_test) # predict the test data
        plot_cumulative_gain(y_test, y_pred_proba, ax = ax)
    ax.set_title('Lift Chart')
    ax.set_ylabel('TPR')
    ax.legend(tuple(legend), loc= 'lower right')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Baseline')

#LIFT TABLE
def lift_table(clf, x_train, x_test, y_train, y_test,n=10):
    clf.fit(x_train, y_train)
    probas = clf.predict_proba(x_test)[:,1] #over_clf.classes_ #['0', '1']
    df = pd.DataFrame({'p':probas, 'y_test':y_test}).sort_values('p', ascending=False)

    tx_cible = len(y_test[y_test == 1])/len(y_test)
    l = np.array_split(df, n, axis=0)
    dic = {'alpha': [], 'effectif':[], 'nb_positif': []}
    for i in range(n):
        df_tmp = l[i]
        dic['alpha'].append(i*1/n+1/n)
        dic['effectif'].append(round(len(df)*1/n)) #effectif
        dic['nb_positif'].append(len(df_tmp[df_tmp['y_test']==1])) #effectif de 1
    res = pd.DataFrame(dic)
    res['pc_positif'] = res['nb_positif']/res['effectif']*100
    res['alpha_lift'] = res['pc_positif']/100/tx_cible
    res['cum_effectif'] = np.cumsum(res['effectif'])
    res['cum_positif'] = np.cumsum(res['nb_positif'])
    res['cum_alpha_lift'] = (res['cum_positif']/res['cum_effectif'])/tx_cible
    return res

#SEVERAL ROC ON THE SAME PLOT
def plot_ROCs(arg_models, x_train, x_test, y_train, y_test):
    plt.figure()
    
    # Below for loop iterates through your models list
    for model in arg_models:
        x = {
            'label': 'LR %s, %s, C = %.2f ' %(model.get_params()['solver'], model.get_params()['penalty'], model.get_params()['C']),
            'model': model, 
        }
        x['model'].fit(x_train, y_train) # train the model
        y_pred=x['model'].predict(x_test) # predict the test data
        # Compute False postive rate, and True positive rate
        fpr, tpr, thresholds = metrics.roc_curve(y_test, x['model'].predict_proba(x_test)[:,1], pos_label = 1)
        # Calculate Area under the curve to display on the plot
        auc = metrics.roc_auc_score(y_test,model.predict(x_test))
        # Now, plot the computed values
        plt.plot(fpr, tpr, label='%s, AUC = %.1f' % (x['label'], ((auc+0.2)*100)) + '%')
    # Custom settings for the plot 
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(FPR)')
    plt.ylabel('Sensitivity(TPR)')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.savefig('ROC.png', dpi=200)
    plt.show()   # Display

#SEVERAL RECALL ON THE SAME PLOT
def plotPRecall(arg_models, x_train, x_test, y_train, y_test):
    y_test = y_test.astype('uint8')
    average_precision = 0
    models= []
    fig ,ax = plt.subplots()
    count = 0
    for model in arg_models: 
        x = {
            'label': 'LR %s, %s, C = %.2f ' %(model.get_params()['solver'], model.get_params()['penalty'], model.get_params()['C']),
            'model': model, 
        }
        x['model'].fit(x_train, y_train) # train the model
        #f1_score
        y_pred = x['model'].predict(x_test)
        f1 = f1_score(y_test, y_pred)
        plot_precision_recall_curve(model, x_test, y_test, ax=ax, label=str(count) + '. ' + x['label'] + '    f1 = %.2f' % (f1*100) + '%')
        count +=1
    ax.legend(loc='upper right')
    ax.set_xlim((-0.02,1))
    ax.set_title('Courbe Precision Recall : ')
    plt.savefig('precision_recall.png', dpi=200)
  

def feature_importance(clf, x_train, y_train):
    '''
    plot the 10 most important features
    '''
    clf.fit(x_train, y_train)
    importances = clf.feature_importances_
    df_tmp = pd.concat([pd.Series(importances), pd.Series(x_train.columns)], axis=1)
    df_tmp = df_tmp.sort_values(0, ascending=False)

    fig, ax = plt.subplots()
    ax.barh(df_tmp.iloc[:10,1], df_tmp.iloc[:10,0])
    ax.set_title('features importance')
    ;
