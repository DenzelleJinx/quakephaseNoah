import obspy
from obspy.core import Stream
import seisbench.util as sbu
import pandas as pd
import numpy as np
from .load_MLmodel_LBQ import load_MLmodel
from .xprob import prob_ensemble
from .streamprocess import stfilter, check_compile_stream, expend_trace, sbresample
from .pfinput import load_check_input
from .xpick import get_picks
import gc


def load_models(paras):
    """
    Loads ML models. This method is intended to be called only by the 'apply' method.
    
    Parameters:
    paras (dict): A dictionary formed from calling the method 'safe_load' on a .yaml parameters file.

    Returns:
    phasemodels (list): A list containing each loaded ML model.

    Raises:
    - ValueError: If any of the models were loaded incorrectly.
    """
    phasemodels = []
    for imodel in paras['MLmodel']:  # loop over each model id
        for irescaling in paras['rescaling']:  # loop over each rescaling_rate
            phasemodels.append(load_MLmodel(model_id=imodel, rescaling_rate=irescaling, 
                                            overlap_ratio=paras['overlap_ratio'], blinding=None))  # blinding=(0, 0)
    Nmlmd = len(paras['MLmodel'])
    Nresc = len(paras['rescaling'])  # total number of rescaling rates
    if (len(phasemodels)) != Nmlmd * Nresc:
        raise ValueError("ML models were not loaded properly")
    return phasemodels


def apply(data: Stream, file_para: str):
    """
    Loads and applies phase picking ML models to input data as indicated by the parameters.yaml file's instruction. 

    Parameters:
    - data (Obspy Stream): An obspy Stream object containing AE data.
    - file_para (str): str, path to the paramter YAML file for quakephase;

    Returns:
    - output (dict): Dictionary containg the following keys:
            'prob': obspy stream object, phase probability for each station,
            'pick': list of xpick object, phase picks for each station.

    Raises:
    - ValueError: If input data is not an obspy Stream.
    """



    # load and check input parameters
    paras = load_check_input(file_para=file_para)

    # load ML models
    phasemodels = load_models(paras)

    # format output
    output = {}
    if (paras['output'].lower() == 'prob') or (paras['output'].lower() == 'all'):
        output['prob'] = obspy.Stream()
    if (paras['output'].lower() == 'pick') or (paras['output'].lower() == 'all'):
        output['pick'] = []

    # Ensure input data is an obspy stream
    if not isinstance(data, (obspy.Stream)):
        raise ValueError(f"Input datatype is not an obspy stream, it is: {type(data)}")    

    # get station list in the stream data
    station_list = list(set(itr.stats.station for itr in data))  # remove duplicates

    # apply model to data streams, loop over each station
    for istation in station_list:
        istream = check_compile_stream(data.select(station=istation))  # check and compile stream, maybe can put intp apply_per_station
        ioutput = apply_per_station(istream, phasemodels, paras)

        # append results to output
        if (paras['output'].lower() == 'prob') or (paras['output'].lower() == 'all'):
            output['prob'] += ioutput['prob']
        if (paras['output'].lower() == 'pick') or (paras['output'].lower() == 'all'):
            output['pick'] += ioutput['pick']
        
        # delete istream and ioutput to save memory
        del ioutput
        del istream
        gc.collect()

    if (paras['output'].lower() == 'pick') or (paras['output'].lower() == 'all'):
        output['pick'] = sbu.PickList(sorted(output['pick']))  # sort picks
        
        

        # format pick output to specified format
        if paras['pick']['format'] is None:
            pass
        elif paras['pick']['format'].lower() == 'list':
            # output is list of pick_dict
            output['pick'] = [ipick.__dict__ for ipick in output['pick']]  # convert to dict
        elif paras['pick']['format'].lower() == 'dict':
            # output is dict of pick_list
            output['pick'] = [ipick.__dict__ for ipick in output['pick']]  # convert to dict
            output['pick'] = {k: [d[k] for d in output['pick']] for k in output['pick'][0]}
        elif paras['pick']['format'].lower() == 'dataframe':
            # output is pick dataframe
            output['pick'] = [ipick.__dict__ for ipick in output['pick']]  # convert to dict
            output['pick'] = pd.DataFrame(output['pick'])

    return output



def apply_per_station(istream, phasemodels, paras):
    """
    Applies loaded ML model to obspy Stream objects containing a single trace pertaining to a specific station.


    Parameters:
    - istream (Stream): obspy Stream object containing data from a single station
    - phasemodels (list): A list containing each loaded ML model.
    - paras (dict): Dictionary containing the following keys:
            'frequency': list of frequency ranges, e.g., [None, [1, 10], [10, 20], [20, 50]],
            'prob_sampling_rate': None or float, sampling rate for the output probability stream,
            'ensemble': str, method for ensemble, 'pca, 'max', 'semblance', ''mean' or 'median',
            'output': str, output type, 'prob', 'pick' or 'all'.
    
    Returns:
    - ioutput (dict):  Dictionary containg the following keys:
                'prob': obspy stream object, phase probability for each station,
                'pick': list of xpick object, phase picks for each station.

    """

    probs_all = []
    for imodel in phasemodels:
        # loop over each model
        predictionWindowLength = imodel.in_samples / float(imodel.sampling_rate)  # prediction window length of the model, in seconds

        for ifreq in paras['frequency']:
            # loop over each frequency range
            # stream_ft = istream.copy()

            # auto expend data if required
            if ('auto_expend' in paras['data']):
                # auto expend data to the required length if input is not enough
                predictionWindowLength_used = predictionWindowLength * paras['data']['auto_expend']['window_ratio']
                trace_expended = False
                itrace_starttime_min = istream[0].stats.starttime
                itrace_endtime_max = istream[0].stats.endtime
                for itrace in range(istream.count()):
                    if istream[itrace].stats.starttime < itrace_starttime_min:
                        itrace_starttime_min = istream[itrace].stats.starttime
                    if istream[itrace].stats.endtime > itrace_endtime_max:
                        itrace_endtime_max = istream[itrace].stats.endtime
                    if (istream[itrace].stats.endtime - istream[itrace].stats.starttime) < predictionWindowLength_used:
                        # need to expend data
                        istream[itrace] = expend_trace(trace=istream[itrace], window_in_second=predictionWindowLength_used, method=paras['data']['auto_expend']['method'])
                        trace_expended = True

            # filter data
            if (isinstance(ifreq, (list))):
                # filter data in specified frequency range
                stfilter(istream, fband=ifreq)

            # obtain phase probability for each model and frequency
            iprob = imodel.annotate(stream=istream)
            if ('auto_expend' in paras['data']) and (trace_expended):
                print('trace was expended!')
                # need to trim probability data to the original length
                iprob.trim(starttime=itrace_starttime_min, endtime=itrace_endtime_max, nearest_sample=True)
            probs_all.append(iprob)
    
    Nfreq = len(paras['frequency'])  # total number of frequency ranges
    assert(len(probs_all)==len(phasemodels)*Nfreq)

    if len(probs_all) == 1:
        prob = probs_all[0]
        if paras['prob_sampling_rate'] is not None:
            # resample prob to the set frequency sampling_rate
            sbresample(stream=prob, sampling_rate=paras['prob_sampling_rate'])
    else:
        # remove potential empty prob_streams
        for iprob in probs_all:
            for itrace in iprob:
                if (itrace.count()==0): iprob.remove(itrace)
        probs_all = [iprob for iprob in probs_all if iprob.count()>0]

        if len(probs_all) > 1:            
            # aggregate results from different models/predictions
            prob = prob_ensemble(probs_all=probs_all, method=paras['ensemble'], sampling_rate=paras['prob_sampling_rate'])
        else:
            prob = probs_all[0]
            if paras['prob_sampling_rate'] is not None:
                # resample prob to the set frequency sampling_rate
                sbresample(stream=prob, sampling_rate=paras['prob_sampling_rate'])

    ioutput = {}
    if (paras['output'].lower() == 'prob') or (paras['output'].lower() == 'all'):
        ioutput['prob'] = prob

    if (paras['output'].lower() == 'pick') or (paras['output'].lower() == 'all'):
        # get picks   
        ioutput['pick'] = get_picks(prob=prob, paras=paras)

        # # check
        # pick_check = phasemodels[0].classify(istream, P_threshold=paras['pick']['P_threshold'],
        #                                      S_threshold=paras['pick']['S_threshold'])
        # assert(len(ioutput['pick'])==len(pick_check.picks))
        # for hh, hhpick in enumerate(ioutput['pick']):
        #     print(pick_check.picks[hh], hhpick)
        #     print()
        #     print(pick_check.picks[hh].__dict__, hhpick.__dict__)
        #     print()
        #     assert(pick_check.picks[hh].__dict__ == hhpick.__dict__)
        #     print('they are the same')

    return ioutput