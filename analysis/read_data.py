import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy.stats import binom_test, norm
from itertools import product
import os

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
            n_resp = np.sum(~pd.isna(thiscond.response)) # n. of actually given responses
            # Log-linear correction (Hautus 1995)
            hitP = (len(thiscond[(thiscond.hit==1)&(thiscond['response']=='j')]) + 0.5)/(n_resp + 1)
            faP = (len(thiscond[(thiscond.hit==0)&(thiscond['response']=='j')]) + 0.5)/(n_resp + 1)
            hitZ = norm.ppf(hitP)
            faZ = norm.ppf(faP)
            dprime = hitZ-faZ
            criterion = -(hitZ + faZ)/2
            datawithdprimes.loc[(datawithdprimes.subject==s)&(datawithdprimes.expected==e), 'dprime'] = dprime
            datawithdprimes.loc[(datawithdprimes.subject==s)&(datawithdprimes.expected==e), 'criterion'] = criterion
    
    return datawithdprimes

# --------------------------------------------------------------------------------------------------------

def make_pretty_plot(avgdata, measure='hit', excl=True, fname=None, saveimg=False):
    
    assert(measure in ['hit', 'dprime', 'criterion'])
    
    # Get all differences
    alldiffs = []
    for sub in avgdata.subject.unique():
        thisdiff = avgdata[(avgdata.subject==sub)&(avgdata.expected==1)][measure].values[0] - \
                   avgdata[(avgdata.subject==sub)&(avgdata.expected==0)][measure].values[0]
        alldiffs.append(thisdiff)
    alldiffs = pd.DataFrame(alldiffs, columns=['difference'])
    
    fig = plt.figure(figsize=(10,10)) # (10, 8)
    
    ax0 = fig.add_subplot(121)
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
        ax0.set(ylim=(0.0, 1.0))
    elif measure=='criterion':
        ax0.set(ylim=(0.0, 1.0))
    
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
            fname.replace('.pdf', '_noexcl.pdf')
    if saveimg:
        if not os.path.isdir('plots'):
            os.mkdir('plots')
        plt.savefig(os.path.join('plots', fname))

# --------------------------------------------------------------------------------------------------------

def subject_exclusion(allsubjdata):
    
    n = 192 # total number of trials
    
    remove_subjs = []
    
    for s in list(allsubjdata.subject.unique()):
        k = allsubjdata[allsubjdata['subject']==s].hit.sum()
        if binom_test(k, n, p=0.5, alternative='greater')>0.05:
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

