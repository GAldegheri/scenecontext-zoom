import pandas as pd
import numpy as np

# --------------------------------------------------------------------------------------------------------

def get_resp(row):

    respdict = eval(row.response)
    assert(len(respdict)==1)

    resp = respdict[list(respdict.keys())[0]]

    return resp

# --------------------------------------------------------------------------------------------------------

def extract_survey(rawdata):

    survey = {}
    
    rawsurvey = rawdata[(rawdata['trial_type'].str.contains('survey'))&(rawdata['trial_name']!='insert_id')]

    survey['sequence_attention'] = get_resp(rawsurvey.iloc[0])
    survey['object_expect'] = get_resp(rawsurvey.iloc[1])
    survey['percent_expected'] = int(get_resp(rawsurvey.iloc[2]))
    survey['comments'] = get_resp(rawsurvey.iloc[3])
    
    p_exp = rawdata.p_exp.iloc[0]
    survey['p_exp'] = p_exp
        
    survey = pd.DataFrame(survey, index=[0])
    
    return survey 

# --------------------------------------------------------------------------------------------------------

def extract_sanitychecks(rawdata):
    
    sanchecks = {}
    
    sanchecks['prolific_id'] = rawdata['subject_id'][0]
    
    sanchecks['time_on_instructions'] = rawdata[rawdata['stimulus'].str.contains('short training session!', na=False)].time_elapsed.values[0]/60000
    
    sanchecks['time_on_breaks'] = np.sum(rawdata[rawdata['stimulus']=='<div style="font-size: 40px; color: white;">Press space to continue</div>'].rt.values/1000)
    
    sanchecks['card_size'] = rawdata[rawdata['trial_type']=='virtual-chinrest'].item_width_px.values[0]
    
    sanchecks['fullscreen_closed'] = np.sum(rawdata['Fullscreen']=='no')
    sanchecks['screen_resized'] = rawdata.window_resolution.nunique()
    
    p_exp = rawdata.p_exp.iloc[0]
    sanchecks['p_exp'] = p_exp
    
    sanchecks = pd.DataFrame(sanchecks, index=[0])
    
    return sanchecks

# --------------------------------------------------------------------------------------------------------    

def cleanupdata(rawdata):
    
    if not 'trial_name' in rawdata.columns: # experiment didn't even start
        return None
    
    # only keep stimuli and responses
    cleandata = rawdata[(rawdata['trial_name']=='stimulus_sequence') | (rawdata['trial_name']=='response_sequence')]
    
    # Did not complete experiment:
    #if len(cleandata[cleandata['training_session']==False]) != 192:
    if len(cleandata) < 408:
        
        return None
    
    else:
        
        p_exp = rawdata.p_exp.iloc[0]
        
        cleandata = cleandata[cleandata['training_session']==False] # remove training session
        
        data = []
        for i in np.arange(0, len(cleandata), 2): # take stimulus sequences only
            
            assert(cleandata.iloc[i].trial_name=='stimulus_sequence')
            
            thisrow = {'scene': cleandata.iloc[i].Scene, 'expected': cleandata.iloc[i].Expected,
                       'initview': cleandata.iloc[i].InitView, 'finalview': cleandata.iloc[i].FinalView,
                       'img_2': cleandata.iloc[i].img_2, 'img_3': cleandata.iloc[i].img_3, 
                       'img_4': cleandata.iloc[i].img_4, 'probe_1': cleandata.iloc[i].probe_1,
                       'probe_2': cleandata.iloc[i].probe_2}
            if 'orient_diff' in cleandata.columns:
                thisrow['int_diff'] = cleandata.iloc[i].orient_diff #abs(cleandata.iloc[i].orient_diff)
            elif 'int_diff' in cleandata.columns:
                thisrow['int_diff'] = cleandata.iloc[i].int_diff #abs(cleandata.iloc[i].int_diff)
                
            thisrow['p_exp'] = p_exp
                
            # Response:
            assert(cleandata.iloc[i+1].trial_name=='response_sequence')
            thisrow['response'] = cleandata.iloc[i+1].response
            thisrow['corr_resp'] = cleandata.iloc[i+1].corr_resp.lower()
            
            # Hit:
            if cleandata.iloc[i+1].response==cleandata.iloc[i+1].corr_resp.lower():
                thisrow['hit'] = 1
            elif pd.isna(cleandata.iloc[i+1].response): # response not given
                thisrow['hit'] = np.nan
            else:
                thisrow['hit'] = 0
            
            # RT:
            if pd.isna(cleandata.iloc[i+1].rt): # response not given
                thisrow['rt'] = np.nan
            else:
                thisrow['rt'] = cleandata.iloc[i+1].rt
            
            data.append(thisrow)
            
        data = pd.DataFrame(data)

        return data
