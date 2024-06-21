from concurrent.futures import ThreadPoolExecutor
import obspy
import seisbench.util as sbu
import pandas as pd
import numpy as np
from obspy import UTCDateTime
from .load_MLmodel import load_MLmodel
from .xprob import prob_ensemble
from .streamprocess import stfilter, check_compile_stream, array2stream, expend_trace, sbresample
from .pfinput import load_check_input
from .xpick import get_picks
import gc


def load_models(paras):
    phasemodels = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for imodel in paras['MLmodel']:
            for irescaling in paras['rescaling']:
                futures.append(executor.submit(load_MLmodel, imodel, irescaling, paras['overlap_ratio'], None))
        for future in futures:
            phasemodels.append(future.result())
    return phasemodels


def apply(data, file_para='parameters.yaml'):
    paras = load_check_input(file_para=file_para)
    phasemodels = load_models(paras)

    output = {'prob': obspy.Stream(), 'pick': []} if paras['output'].lower() == 'all' else {}
    if 'prob' in paras['output'].lower():
        output['prob'] = obspy.Stream()
    if 'pick' in paras['output'].lower():
        output['pick'] = []

    if isinstance(data, (obspy.Stream)):
        stream = data
    elif isinstance(data, (obspy.Trace)):
        stream = obspy.Stream(traces=[data])
    elif isinstance(data, str):
        stream = obspy.read(data)
    elif isinstance(data, list) and all(isinstance(item, str) for item in data):
        stream = obspy.Stream()
        for idata in data:
            stream += obspy.read(idata)
    elif isinstance(data, np.ndarray):
        stream = array2stream(data=data, paras=paras['data'])
        paras['prob_sampling_rate'] = 100
    else:
        raise ValueError(f"Unknown data type: {type(data)}")

    station_list = list(set(itr.stats.station for itr in stream))

    def process_station(istation):
        istream = stream.select(station=istation).copy()
        istream = check_compile_stream(istream)
        return apply_per_station(istream, phasemodels, paras)

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_station, station_list)

    for ioutput in results:
        if 'prob' in output:
            output['prob'] += ioutput['prob']
        if 'pick' in output:
            output['pick'] += ioutput['pick']

    if 'pick' in output:
        output['pick'] = sbu.PickList(sorted(output['pick']))
        if isinstance(data, np.ndarray):
            for jjpick in output['pick']:
                for attr, avalue in vars(jjpick).items():
                    if isinstance(avalue, UTCDateTime):
                        new_avalue = (avalue - UTCDateTime(0)) * 100 + 1
                        setattr(jjpick, attr, new_avalue)
        if paras['pick']['format']:
            if paras['pick']['format'].lower() == 'list':
                output['pick'] = [ipick.__dict__ for ipick in output['pick']]
            elif paras['pick']['format'].lower() == 'dict':
                output['pick'] = {k: [d[k] for d in output['pick']] for k in output['pick'][0]}
            elif paras['pick']['format'].lower() == 'dataframe':
                output['pick'] = pd.DataFrame([ipick.__dict__ for ipick in output['pick']])

    return output


def apply_per_station(istream, phasemodels, paras):
    probs_all = []
    for kmodel in phasemodels:
        pdtw = kmodel.in_samples / float(kmodel.sampling_rate)
        for ifreq in paras['frequency']:
            stream_ft = istream.copy()
            if 'auto_expend' in paras['data']:
                pdtw_used = pdtw * paras['data']['auto_expend']['window_ratio']
                trace_expended = False
                itrace_starttime_min = min(stream_ft, key=lambda x: x.stats.starttime).stats.starttime
                itrace_endtime_max = max(stream_ft, key=lambda x: x.stats.endtime).stats.endtime
                for jjtr in range(stream_ft.count()):
                    if (stream_ft[jjtr].stats.endtime - stream_ft[jjtr].stats.starttime) < pdtw_used:
                        stream_ft[jjtr] = expend_trace(trace=stream_ft[jjtr], window_in_second=pdtw_used, method=paras['data']['auto_expend']['method'])
                        trace_expended = True
            if isinstance(ifreq, list):
                stfilter(stream_ft, fband=ifreq)
            kprob = kmodel.annotate(stream=stream_ft)
            if 'auto_expend' in paras['data'] and trace_expended:
                kprob.trim(starttime=itrace_starttime_min, endtime=itrace_endtime_max, nearest_sample=True)
            probs_all.append(kprob)
            del stream_ft
            gc.collect()

    Nfreq = len(paras['frequency'])
    assert len(probs_all) == len(phasemodels) * Nfreq

    if len(probs_all) == 1:
        prob = probs_all[0]
        if paras['prob_sampling_rate']:
            sbresample(stream=prob, sampling_rate=paras['prob_sampling_rate'])
    else:
        probs_all = [iprob for iprob in probs_all if iprob.count() > 0]
        if len(probs_all) > 1:
            prob = prob_ensemble(probs_all=probs_all, method=paras['ensemble'], sampling_rate=paras['prob_sampling_rate'])
        else:
            prob = probs_all[0]
            if paras['prob_sampling_rate']:
                sbresample(stream=prob, sampling_rate=paras['prob_sampling_rate'])

    ioutput = {}
    if 'prob' in paras['output'].lower():
        ioutput['prob'] = prob
    if 'pick' in paras['output'].lower():
        ioutput['pick'] = get_picks(prob=prob, paras=paras)

    return ioutput
