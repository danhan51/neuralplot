# Functions for extracting relevant data from behavioral files
import pandas as pd
import numpy as np
import os

EVENT_CODES_TO_EVENT_NAME = {
    9: 'trial_start',
    18: 'trial_end_ml2', #trial end by ml2 (after blue idle screen flashes)
    10: 'fix_cue',
    20: 'sample_on',
    21: 'sample_off',
    49: 'timeout',
    50: 'rew', 
    51: 'trial_end_blue' #trial ends, and blue idle screen flashes
}

def loadQuad(animal, date):
    """
    loads quad object for given animal and date
    NOTE: Here is where to change dir paths to match your system :)
    """
    if animal == 'Diego':
        subject = 'S1'
    elif animal == 'Pancho':
        subject = 'S2'


    basedir = '/home/danhan/code/data/quad_data'
    paths = {
    'ml2_dir': f'{basedir}/{animal}/{date}',
    'conditions_dir': f'{basedir}/{animal}/conditions_quadrilaterals_{subject}.txt', 
    'tdt_dir': f'/home/danhan/freiwaldDrive/ltian/recordings/{animal}/{date}',
    'spikes_dir': f''
    }

    return Quads(paths)

class Quads:
    def __init__(self, paths):
        self.paths = paths
        """
        format (if not use load function):
            paths = {
            'ml2_dir': f'{basedir}/{animal}/{date}',
            'conditions_dir': f'{basedir}/{animal}/conditions_quadrilaterals_{subject}.txt', 
            'tdt_dir': f'/home/danhan/freiwaldDrive/ltian/recordings/{animal}/{date}',
            'spikes_dir': f''
            }
        """

        self.ml2_dat_list = self.loadML2Data()              # dict with all trial info (convert bhv2 to mat)
        self.conditions = self.loadCondtionsFile() # conditions file loaded from text as pd df
        self.tdt_dat_list = self.loadTdtData()
        self.spike_times = self.loadSpikeTimes()


        self.prettyBeh = self.generatePrettyBehDF()
        self.prettyNeural = self.generatePrettyNeuralDF()


    
    def loadML2Data(self):
        """
        Function to load ml2 data, returns list of beh dat structs. Pretty function will concat
        """
        from scipy.io import loadmat

        data_dir = self.paths['ml2_dir']
        bhv_list = [f'{data_dir}/{f}' for f in os.listdir(data_dir) if f.endswith('mat')]
        beh_dat_list = []
        #make sure list is ordered by session properly
        bhv_list_ordered = []
        for i in range(1,len(bhv_list)+1):
            for bhv in bhv_list:
                if bhv.split('.')[0].endswith(str(i)):
                    bhv_list_ordered.append(bhv)
        for bhv in bhv_list_ordered:
            beh_dat_list.append(loadmat(bhv,simplify_cells=True))
        return beh_dat_list

    def loadTdtData(self, load_eye_tracking = False):
        """
        Function to load tdt data, will concat sessions
        """
        from tdt import read_block
        tdt_list = os.listdir(self.paths['tdt_dir'])
        tdt_dat_list = []
        tdt_sess_durations = []
        evs_load = ['streams', 'epocs']
        stores_load = ['SMa1', 'Rew_','PhD2']
        if load_eye_tracking:
            stores_load.append('Eyee')
        #probably overcomplicated but wnated ot make sure it does what I want
        #make sure tdt list sessions are ordered properly in time
        tdt_list_ordered = []
        sess_times = sorted(set([int(s.split('-')[-1]) for s in tdt_list]))
        for sess_time in sess_times:
            for sess in tdt_list:
                if sess_time == int(sess[-6:]):
                    tdt_list_ordered.append(sess)
        
        for tdt_session in tdt_list_ordered:
            fullpath = f"{self.paths['tdt_dir']}/{tdt_session}"
            session_data = read_block(fullpath,
                                    evtype=evs_load,
                                    store=stores_load,t1=0,t2=30)
            tdt_dat_list.append(session_data)
            tdt_sess_durations.append(session_data.info.duration.total_seconds())
        return tdt_dat_list

    def loadCondtionsFile(self):
        """
        Function to load the condtions file as df for getting stim names
        """
        with open(self.paths['conditions_dir'], 'r') as f:
            conditions = pd.read_csv(f, delimiter = '\t')
        return conditions
    
    def loadSpikeTimes(self):
        """
        Function to load spike times
        """

    def generatePrettyBehDF(self):
        """
        Flattens beh data into something workable
        """
        df_columns = ['trial_ml2','stim_index','stim_name','fixation_success_binary']
        df = pd.DataFrame(columns = df_columns)
        stim_index = 0 
        for session_index, dat in enumerate(self.ml2_dat_list):
            trial_nums = [int(t.split('Trial')[1]) for t in dat.keys() if (t.startswith('Trial') and t != 'TrialRecord')]
            for trial in trial_nums:
                stim,success_fail = self.getWhatStimEachPresentation(session_index,trial)
                for stim,success in zip(stim,success_fail):
                    new_entry = pd.DataFrame([
                        {
                            'beh_session': session_index,
                            'block_num': dat[f'Trial{trial}']['Block'],
                            'trial_ml2':trial, #relative to session
                            'stim_index': stim_index, #unique index for each stim presentation, counts over sessions
                            'condition': dat[f'Trial{trial}']['Condition'],
                            'stim_name':stim,
                            'fixation_success_binary':success
                        }
                    ])
                    df = pd.concat([df,new_entry], ignore_index=True)
                    stim_index += 1
        return df
    def generatePrettyNeuralDF(self):
        """
        Same as beh
        """
        
        df_columns = ['trial_ml2','stim_index','code_type','on','off'] #trial calculated by num 9's
        full_df = pd.DataFrame(columns = df_columns)
        
        stim_counter = 0
        for session_ind, session_dat in enumerate(self.tdt_dat_list):
            session_df = pd.DataFrame(columns = df_columns)
            beh_codes = session_dat.epocs.SMa1.data
            ons = session_dat.epocs.SMa1.onset
            offs = session_dat.epocs.SMa1.offset
            trial_counter = 0
            for code,on,off in zip(beh_codes,ons,offs):
                if code in range(102,132):
                    continue
                elif code == 9:
                    trial_counter += 1
                    stim_index = np.nan
                elif code == 20:
                    stim_index = stim_counter
                    stim_counter += 1
                else:
                    stim_index = np.nan
                code_type = EVENT_CODES_TO_EVENT_NAME[code]
                new_entry = pd.DataFrame([
                    {
                        'trial_ml2': trial_counter,
                        'stim_index': stim_index,
                        'code_type': code_type,
                        'on': on,
                        'off': off
                    }
                ])
                session_df = pd.concat([session_df,new_entry], ignore_index=True)

            session_df = self.assignEventMarkerstoPDTimes(session_ind, session_df)
            full_df = pd.concat([full_df,session_df], ignore_index=True)
        return full_df

    def getEphysStreamTrange(self,trange):
        """
        Probbaly dont need this, maybe for LFP?
        Pull out ephys data for given time range
        RSn2 = ch 1-256
        RSn3 = ch 257-512
        trange = tuple (start,end) seconds
        """
        n = self.tdt_dat

        rs2_streams = np.array(n.streams.RSn2.data)
        rs2_channels = np.array(n.streams.RSn2.channels)
        rs2_fs = n.streams.RSn2.fs
        rs2_start = n.streams.RSn2.start_time

        rs3_streams = np.array(n.streams.RSn3.data)
        rs3_channels = np.array(n.streams.RSn3.channels)
        rs3_fs = n.streams.RSn3.fs
        rs3_start = n.streams.RSn3.start_time

        assert rs2_start == rs3_start == 0, 'why neural start time not 0?'
        assert rs2_fs == rs3_fs, 'why neural fs different?'

        fs = rs2_fs

        sample_start = int(np.floor(trange[0]*fs))
        sample_end = int(np.ceil(trange[1]*fs))
        stream_times = np.linspace(sample_start/fs,sample_end/fs,sample_end-sample_start + 1)
        #for sanity
        assert stream_times[0] <= trange[0]
        assert stream_times[-1] >= trange[-1]


        data_by_channel_dict = {}
        data_by_channel_dict['time'] = stream_times
        for channel in range (1,513):
            if channel < 257:
                stream = rs2_streams[np.where(rs2_channels == channel)[0][0],sample_start:sample_end + 1]
                data_by_channel_dict[channel] = stream
            elif channel >= 257:
                stream = rs3_streams[np.where(rs2_channels == (channel - 256))[0][0],sample_start:sample_end + 1]
                data_by_channel_dict[channel] = stream
        print(data_by_channel_dict.keys())
        return data_by_channel_dict

    def assignEventMarkerstoPDTimes(self, session, neural):
        """
        called internally
        Adds column to pretty neural to assign photodiode times to relevant events
        """

        def nearest_value(arr, target, max_dist=None):
            """
            Return (index, value) of the array element closest to target.
            If max_dist is given, return np.nan if no value is within that distance (i.e. no pd trigger for event).
            """

            d = np.abs(arr - target)

            idx = np.argmin(d)
            val = arr[idx]
            dist = d[idx]

            # Check distance constraint
            if max_dist is not None and dist > max_dist:
                return np.nan, np.nan

            return idx, val
        
        session_time_offset = 0
        if session > 0:
            for i in range(session):
                session_time_offset += self.sessionDurations[i]
        
        pd_times = self.getPhotodiodeThresholdCrossings(session)

        max_dist = 0.1 #thresh for how far pd time can be from 'onset' time
        neural['photodiode_time'] = np.nan

        rew_times = self.tdt_dat_list[session].epocs.Rew_
        

        rew_inds_taken = []
        inds_taken = []
        for i, row in neural.iterrows():
            if row['code_type'] in ['trial_start','trial_end_ml2']:
                #if no pd use avg of on/off signal time
                pd_time = np.mean([row['on'],row['off']])
            elif row['code_type'] == 'rew':
                idx,pd_time = nearest_value(rew_times, row['on'], max_dist = max_dist)
                assert idx not in rew_inds_taken, 'on time two rew? no good'
                rew_inds_taken.append(idx)
            else:
                idx, pd_time = nearest_value(pd_times, row['on'], max_dist = max_dist)
                if idx in inds_taken:
                    if row['on'] - neural.loc[i-1,'on'] > 0.1:
                        #some events happen to fast for pd to respond to both?
                        #like sample off and fix cue on for success trials
                        print(i)
                        print(row['code_type'])
                        assert False
                inds_taken.append(idx)

            neural.loc[i, 'photodiode_time'] = session_time_offset + pd_time
        #some sanity checks, all times assigned
        assert len(rew_inds_taken) == len(rew_times), 'not all rews assigned, why?'
        assert len(set(inds_taken)) == pd_times, 'why not all pd times used'

        return neural

            
    
    def getPhotodiodeThresholdCrossings(self, session):
        """
        Gets photodiode trigger times to align with neural events later
        """
        from scipy.signal import butter,lfilter,freqz

        pd_analog = self.tdt_dat_list[session].streams.PhD2.data

        fs = self.tdt_dat_list[session].streams.PhD2.fs
        cutoff_freq = 60
        nyquist_freq = 0.5*fs
        normal_cutoff = cutoff_freq/nyquist_freq
        order = 4

        b,a = butter(order,normal_cutoff,btype='low',analog=False)

        pd_filt = lfilter(b,a,pd_analog)

        inds = np.array(list(range(0,len(pd_filt))))
        times = inds/fs

        assert len(pd_filt) == len(times) #idk just in case

        min_val = np.percentile(pd_filt,25)
        max_val = np.percentile(pd_filt,75)
        thresh = (min_val+max_val)/2


        inds_crossings = np.where(
            ((pd_filt[:-1] <= thresh) & (pd_filt[1:] > thresh)) |    
            ((pd_filt[:-1] > thresh) & (pd_filt[1:] <= thresh))     
        )[0]

        crossing_times = times[inds_crossings]

        return crossing_times

    def getListStimNames(self, session, trial_ml2):
        """
        get list of stim file names for given trial.

        inputs:
        trial (int): monkeylogic (1 indexed) trial num
        """

        dat_trial = self.ml2_dat_list[session][f'Trial{trial_ml2}']
        condition_num = dat_trial['Condition']
        conds = self.conditions
        stim_list = []
        for i in range (2,32):
            stim_full = conds.loc[conds['Condition'] == condition_num, f'TaskObject#{i}'].iloc[0]
            stim = stim_full.split('/')[1].split(')')[0]
            stim_list.append(stim)
        return stim_list
    
    def getWhatStimEachPresentation(self, session, trial_ml2):
        """
        Get list of stims on each presentation.
        in:
        trial (int): 1 index trial
        ret:
        stim_each_present (list): stim name on each presentation
        stim_success_fail (list): True is fixated, False otherwise
        """

        dat_trial = self.ml2_dat_list[session][f'Trial{trial_ml2}']
        stim_list = self.getListStimNames(session,trial_ml2)
        beh_codes = dat_trial['BehavioralCodes']['CodeNumbers']
        stim_codes = [c%100 for c in beh_codes if 102 <= c <= 131]
        stim_success_fail = [c != stim_codes[i+1] for i,c in enumerate(stim_codes) if i < len(stim_codes)-1]
        stim_each_present = [stim_list[c-2] for c in stim_codes]
        if len(stim_each_present) > 0:
            stim_success_fail.append(True) #last fix true bc trial not end otherwise
        assert len(stim_success_fail) == len(stim_each_present), 'why diff lens'

        return stim_each_present, stim_success_fail
    def AlignBehWithNeuralData(self, trial_ml2):
        """
        Finds neural on/off times aligned to this beh trial
        """
        neural_beh_codes = self.tdt_dat.epocs.SMa1.data
        neural_beh_codes_times = self.tdt_dat.epocs.SMa1.onset
        assert len(neural_beh_codes) == len(neural_beh_codes_times), 'why diff lengths'
        start_counter = 0
        start_time = None
        end_time = None
        found_start = False
        for i,code in neural_beh_codes:
            if code == 9:
                start_counter += 1
            if start_counter == trial_ml2:
                start_time == neural_beh_codes_times[i]
                found_start = True
            if code == 18 and found_start:
                end_time = neural_beh_codes_times[i]

        assert start_time is not None and end_time is not None

        return (start_time,end_time)
    


        







