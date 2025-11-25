import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy.stats import binomtest, norm
from itertools import product
import os

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter

# --------------------------------------------------------------------------------------------------------

def prepare_data_for_glmm(allsubjdata):
    
    # remove trials with missing responses
    allsubjdata = allsubjdata.dropna(subset = ['response'])
    
    # label trials as same or different based on the correct response key
    allsubjdata['sameordiff'] = allsubjdata['corr_resp'].map({'f': 'same', 'j': 'different'})
    
    # code 'same' responses as 0 and 'different' as 1
    allsubjdata['response_samediff'] = allsubjdata['response'].map({'f': 0, 'j': 1})
    
    # set stimulus intensity difference to 0 on 'same' trials
    allsubjdata.loc[allsubjdata['sameordiff']=='same','int_diff'] = 0.0
    allsubjdata['abs_diff'] = allsubjdata['int_diff'].abs() 
    
    # make 'expected' and 'p_exp' categorical variables
    allsubjdata['expected'] = allsubjdata['expected'].astype('category')
    allsubjdata['p_exp'] = (allsubjdata['p_exp'] * 100).astype(int).astype('category')
    
    return allsubjdata

# --------------------------------------------------------------------------------------------------------

def glmm_analysis_r(allsubjdata):
    
    # move data to R
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        ro.globalenv['r_data'] = allsubjdata
        
    # run the analysis in R
    ro.r('''
        # Load library inside R
        library(lme4)
        library(emmeans)
        
        # Make p_exp a factor with "75" group as reference
        r_data$p_exp <- as.factor(r_data$p_exp)
        r_data$p_exp <- relevel(r_data$p_exp, ref = "75")
        
        # Fit the model
        # We use the dataframe 'r_data' we just pushed
        model <- glmer(response_samediff ~ abs_diff * expected * p_exp + 
                    (1 + abs_diff * expected | subject),
                    data = r_data,
                    family = binomial(link = "probit"),
                    # Control settings often help with complex models
                    control = glmerControl(optimizer = "bobyqa"))
        
        # Extract the summary and coefficient matrix
        # We save these to R variables 'summ' and 'coef_mat'
        summ <- summary(model)
        coef_mat <- coef(summ)
        
        # Get sensitivity (Slopes of abs_diff)
        sens_trends <- emtrends(model, ~ expected * p_exp, var = "abs_diff")
        sens_df <- as.data.frame(sens_trends)
        
        # Get criterion (intercept/bias)
        # criterion c = -1 * (Probit Score)
        crit_means <- emmeans(model, ~ expected * p_exp, at = list(abs_diff = 0))
        crit_df = as.data.frame(crit_means)
        
        # Get sensitivity contrasts
        sens_contrasts <- pairs(sens_trends, simple = "expected")
        
        # Get criterion contrasts
        crit_contrasts <- pairs(crit_means, simple = "expected")
        
        # The coef() function extracts the sum of (Fixed Effects + Random Effects) 
        # for every subject.
        subject_coefs <- coef(model)$subject
    ''')
    
    # retrieve the results back into Python
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        # A. Get the numbers (This becomes a numpy array automatically)
        data_array = ro.globalenv['coef_mat']
        
        # B. Get the names explicitly as Python lists
        # We call R functions 'rownames' and 'colnames' on the object inside R
        row_names = list(ro.r('rownames(coef_mat)'))
        col_names = list(ro.r('colnames(coef_mat)'))
        
        sensitivity_plot_data = ro.globalenv['sens_df']
        sens_row_names = list(ro.r('rownames(sens_df)'))
        sens_col_names = list(ro.r('colnames(sens_df)'))
        
        criterion_plot_data = ro.globalenv['crit_df']
        crit_row_names = list(ro.r('rownames(crit_df)'))
        crit_col_names = list(ro.r('colnames(crit_df)'))
        
        subj_df = ro.globalenv['subject_coefs']
        subj_row_names = list(ro.r('rownames(subject_coefs)'))
        subj_col_names = list(ro.r('colnames(subject_coefs)'))

    # 4. Construct the DataFrame manually
    results_df = pd.DataFrame(data_array, 
                            index=row_names, 
                            columns=col_names)

    sensitivity_plot_data = pd.DataFrame(sensitivity_plot_data,
                                        index=sens_row_names,
                                        columns=sens_col_names)
    criterion_plot_data = pd.DataFrame(criterion_plot_data,
                                    index=crit_row_names,
                                    columns=crit_col_names)
    subj_df = pd.DataFrame(subj_df,
                        index=subj_row_names,
                        columns=subj_col_names)

    sensitivity_plot_data.rename(columns={'abs_diff.trend': 'Sensitivity_Slope', 
                                        'SE': 'Standard_Error'}, inplace=True)

    # Criterion: The column 'emmean' is the Intercept. 
    # We must flip the sign to get 'c'
    criterion_plot_data['Criterion_c'] = -1 * criterion_plot_data['emmean']
    criterion_plot_data['Standard_Error'] = criterion_plot_data['SE'] 
    
    return results_df, sensitivity_plot_data, criterion_plot_data, subj_df

# --------------------------------------------------------------------------------------------------------

def get_surveys(allsubjdata):
    
    whichsubjs = allsubjdata.subject.unique()
    
    allsurveys = pd.read_csv('data/allsurveys.csv')
    
    allsurveys = allsurveys[allsurveys['subject'].isin(whichsubjs)]
    
    return allsurveys
    
# --------------------------------------------------------------------------------------------------------

def compute_dprimes(allsubjdata):
    
    datawithdprimes = allsubjdata.copy()
    datawithdprimes['dprime'] = np.nan
    datawithdprimes['criterion'] = np.nan
    
    # Get hits and FAs
    for s in allsubjdata.subject.unique():
        thissubj = allsubjdata[allsubjdata['subject']==s]
        #assert(len(thissubj)==192)
        for e in [0, 1]: # expected, unexpected
            thiscond = thissubj[thissubj['expected']==e]
            #n_resp = np.sum(~pd.isna(thiscond.response)) # n. of actually given responses
            n_signal = len(thiscond[(thiscond['corr_resp']=='j')&(~pd.isna(thiscond.response))])
            n_noise = len(thiscond[(thiscond['corr_resp']=='f')&(~pd.isna(thiscond.response))])
            # Log-linear correction (Hautus 1995)
            hitP = (len(thiscond[(thiscond.hit==1)&(thiscond['response']=='j')]) + 0.5)/(n_signal + 1)
            faP = (len(thiscond[(thiscond.hit==0)&(thiscond['response']=='j')]) + 0.5)/(n_noise + 1)
            hitZ = norm.ppf(hitP)
            faZ = norm.ppf(faP)
            dprime = hitZ-faZ
            criterion = -(hitZ + faZ)/2
            datawithdprimes.loc[(datawithdprimes.subject==s)&(datawithdprimes.expected==e), 'dprime'] = dprime
            datawithdprimes.loc[(datawithdprimes.subject==s)&(datawithdprimes.expected==e), 'criterion'] = criterion
    
    return datawithdprimes

# --------------------------------------------------------------------------------------------------------

def make_pretty_plot(avgdata, measure='hit', excl=True, fname=None, 
                     cloudplot=False, saveimg=False):
    
    assert(measure in ['hit', 'dprime', 'criterion'])
    
    # Get all differences
    alldiffs = []
    for sub in avgdata.subject.unique():
        thisdiff = avgdata[(avgdata.subject==sub)&(avgdata.expected==1)][measure].values[0] - \
                   avgdata[(avgdata.subject==sub)&(avgdata.expected==0)][measure].values[0]
        alldiffs.append(thisdiff)
    alldiffs = pd.DataFrame(alldiffs, columns=['difference'])
    
    if cloudplot:
        fig = plt.figure(figsize=(10,10)) # (10, 8)
        ax0 = fig.add_subplot(121)
    else:
        fig = plt.figure(figsize=(7,10))
        ax0 = fig.add_subplot(111)
    
    sns.barplot(x='expected', y=measure, data=avgdata, ci=68, order=[1.0, 0.0], 
                palette='Set2', ax=ax0, errcolor='black', edgecolor='black', linewidth=2, capsize=.2)
    if measure=='hit':
        ax0.set_ylabel('Accuracy', fontsize=30)
    elif measure=='dprime':
        ax0.set_ylabel('d\'', fontsize=30)
    elif measure=='criterion':
        ax0.set_ylabel('Criterion', fontsize=30)
    plt.yticks(fontsize=24) 
    ax0.tick_params(axis='y', direction='out', color='black', length=10, width=2)
    ax0.tick_params(axis='x', length=0, pad=15)
    ax0.set_xlabel(None)
    ax0.set_xticklabels(['Cong.', 'Incong.'], fontsize=30)
    ax0.spines['left'].set_linewidth(2)
    ax0.spines['bottom'].set_linewidth(2)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    if measure=='hit':
        ax0.set(ylim=(0.5, 0.75))
    elif measure=='dprime':
        ax0.set(ylim=(0.0, 1.4))
    elif measure=='criterion':
        ax0.set(ylim=(-0.1, 0.4))
        ax0.axhline(0.0, color='black', linewidth=2, linestyle='--')
    
    if cloudplot:
        ax1 = fig.add_subplot(122)
        sns.violinplot(y='difference', data=alldiffs, color=".8", inner=None)
        sns.stripplot(y='difference', data=alldiffs, jitter=0.07, ax=ax1, color='black', alpha=.5)
        # Get mean and 95% CI:
        meandiff = alldiffs['difference'].mean()
        tstats = pg.ttest(alldiffs['difference'], 0.0)
        ci95 = tstats['CI95%'][0]
        for tick in ax1.get_xticks():
            ax1.plot([tick-0.1, tick+0.1], [meandiff, meandiff],
                        lw=4, color='k')
            ax1.plot([tick, tick], [ci95[0], ci95[1]], lw=3, color='k')
            ax1.plot([tick-0.03, tick+0.03], [ci95[0], ci95[0]], lw=3, color='k')
            ax1.plot([tick-0.03, tick+0.03], [ci95[1], ci95[1]], lw=3, color='k')
        ax1.axhline(0.0, linestyle='--', color='black')
        plt.yticks(fontsize=24) 
        if measure=='hit':
            ax1.set_ylabel('Δ Accuracy', fontsize=30)
            if excl:
                ax1.set(ylim=(-0.2, 0.4))
            else:
                ax1.set(ylim=(-0.3, 0.4))
        elif measure=='dprime':
            ax1.set_ylabel('Δ d\'', fontsize=30)
            ax1.set(ylim=(-2., 2.))
        elif measure=='criterion':
            ax1.set_ylabel('Δ Criterion', fontsize=30)
            ax1.set(ylim=(-1.0, 1.25))
        ax1.axes_style = 'white'
        ax1.tick_params(axis='y', direction='out', color='black', length=10, width=2)
        ax1.tick_params(axis='x', length=0)
        ax1.spines['left'].set_linewidth(2)
        ax1.spines['bottom'].set_linewidth(2)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['top'].set_visible(False)
    plt.tight_layout()
    whichprob = avgdata.p_exp[0]
    if not fname:
        fname = f'p{whichprob*100:g}_{measure}.pdf'
        if not excl:
            fname = fname.replace('.pdf', '_noexcl.pdf')
    if saveimg:
        if not os.path.isdir('plots'):
            os.mkdir('plots')
        plt.savefig(os.path.join('plots', fname))

# --------------------------------------------------------------------------------------------------------

def subject_exclusion(allsubjdata):
    
    n = 192 # total number of trials
    
    remove_subjs = []
    
    for s in list(allsubjdata.subject.unique()):
        k = allsubjdata[allsubjdata['subject']==s].hit.sum().astype('int')
        if binomtest(k, n, p=0.5, alternative='greater').pvalue>0.05:
            remove_subjs.append(s)
    
    return remove_subjs

# --------------------------------------------------------------------------------------------------------

def make_align_plot_together(avgdata, measure='hit', saveimg=False):
    '''
    Collapse across experiments, show results for aligned/misaligned,
    congruent/incongruent (not just cong-incong differences)
    '''
    
    # Get all differences for each p(exp) and alignment
    alldiffs = []
    for (pe, al) in product(avgdata.p_exp.unique(), avgdata.aligned.unique()):
        thiscond = avgdata[(avgdata.p_exp==pe)&(avgdata.aligned==al)]
        for sub in thiscond.subject.unique():
            thisdiff = thiscond[(thiscond.subject==sub)&(thiscond.expected==1)][measure].values[0] - \
                       thiscond[(thiscond.subject==sub)&(thiscond.expected==0)][measure].values[0]
            alldiffs.append({'subject': sub, 'p_exp': pe, 'aligned': al, 'difference': thisdiff})
    alldiffs = pd.DataFrame(alldiffs)
    
    avgdata = avgdata.groupby(['subject', 'expected', 'aligned']).mean().reset_index()
    pal = [[list(sns.color_palette('Paired'))[0], list(sns.color_palette('Paired'))[6]],
           [list(sns.color_palette('Paired'))[1], list(sns.color_palette('Paired'))[7]]]
    
    with sns.axes_style('white'):
        fig = plt.figure(figsize=(7, 6))
        plt.rcParams['ytick.left'] = True
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['axes.linewidth'] = 1.5
        ax = fig.add_subplot(1, 1, 1)
        sns.barplot(x='aligned', y=measure, hue='expected', data=avgdata, 
                    ci=68, order=[True, False], hue_order=[1.0, 0.0],
                    palette='gray', #palette='Paired', 
                    ax=ax, errcolor='black', 
                    edgecolor='black', linewidth=2, capsize=.2)
        for bar_group, p in zip(ax.containers, pal):
            for bar, col in zip(bar_group, p):
                bar.set_facecolor(col)

        ax.legend_.remove()
        
        if measure=='hit':
            ylab = 'Accuracy'
            ax.set(ylim=(0.5, 0.8))
        elif measure=='dprime':
            ylab = 'd\''
            ax.set(ylim=(0.0, 1.0))
        elif measure=='criterion':
            ylab = 'Criterion'
            ax.set(ylim=(0.0, 1.0))
        ax.set_ylabel(ylab, fontsize=20)
        plt.yticks(fontsize=16)
        ax.set_xlabel(None)
        ax.set_xticklabels(['Aligned', 'Misaligned'], fontsize=20)
        ax.yaxis.set_tick_params(width=1.5, length=5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.margins(x=0.08)
    if saveimg:
        if not os.path.isdir('plots'):
            os.mkdir('plots')
        figname = os.path.join('plots', f'aligned_misaligned_together_{measure:s}.pdf')
        plt.savefig(figname)

