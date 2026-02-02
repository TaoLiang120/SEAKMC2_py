import time

import numpy as np

import seakmc.datasps.PostSPS as postSPS
import seakmc.datasps.PreSPS as preSPS
import seakmc.datasps.SaddlePointSearch as mySPS
import seakmc.general.DataOut as Dataout
from seakmc.restart.Restart import RESTART
from seakmc.spsearch.SaddlePoints import Data_SPs

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


def data_find_saddlepointsearch(iav, idav, thisAV,
                                local_coords, thisSOPs, isPGSYMM, thisVNS, isRecycled, Pre_Disps,
                                thisnspsearch, SNC, CalPref, dynmatAV,
                                istep, thissett, seakmcdata, DefectBank_list, thisSuperBasin, Eground,
                                thisDataSPs, thisAVitags, df_delete_SPs, thisundo_idavs, thisfinished_AVs,
                                simulation_time,
                                object_dict, Precursor=False, insituGSPS=False):
    float_precision = thissett.system['float_precision']
    out_paths = object_dict['out_paths']
    LogWriter = object_dict['LogWriter']
    DFWriter = object_dict['DFWriter']
    thiscolor = 0
    ticav = time.time()
    if thisAV is not None:
        if Precursor:
            x = seakmcdata.precursors.at[idav, 'xsn']
            y = seakmcdata.precursors.at[idav, 'ysn']
            z = seakmcdata.precursors.at[idav, 'zsn']
            logstr = f"Precursor ID: {idav} nactive:{thisAV.nactive} nbuffer:{thisAV.nbuffer} nfixed:{thisAV.nfixed}"
            logstr += "\n" + (f"Precursor center fractional coords: "
                              f"{round(x, 5), round(y, 5), round(z, 5)}")
        else:
            x = seakmcdata.defects.at[idav, 'xsn']
            y = seakmcdata.defects.at[idav, 'ysn']
            z = seakmcdata.defects.at[idav, 'zsn']
            logstr = f"ActVol ID: {idav} nactive:{thisAV.nactive} nbuffer:{thisAV.nbuffer} nfixed:{thisAV.nfixed}"
            logstr += "\n" + (f"AV center fractional coords: "
                              f"{round(x, 5), round(y, 5), round(z, 5)}")
        LogWriter.write_data(logstr)

        thisSPS = preSPS.initialize_thisSPS(idav, local_coords, thisnspsearch, thissett)
        thisSPS, df_delete_SPs = mySPS.saddlepoint_search(thiscolor, istep, thissett, idav, thisAV, local_coords,
                                                          thisSOPs, dynmatAV, SNC, CalPref,
                                                          thisSPS, Pre_Disps, thisnspsearch, thisVNS, df_delete_SPs,
                                                          object_dict, Precursor=Precursor, insituGSPS=insituGSPS)

        thisSPS, df_delete_SPs = postSPS.SPs_1postprocessing(thissett, thisSPS, df_delete_SPs, DFWriter,
                                                             nSPstart=thisSPS.nSP, insituGSPS=insituGSPS)
        if Precursor:
            thisSPS, df_delete_this = thisDataSPs.check_dup_avSP(idav, thisSPS, seakmcdata.precursor_neighbors,
                                                                 thisAVitags, thissett.saddle_point)
        else:
            thisSPS, df_delete_this = thisDataSPs.check_dup_avSP(idav, thisSPS, seakmcdata.de_neighbors, thisAVitags,
                                                                 thissett.saddle_point)
        df_delete_SPs = preSPS.update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter)

        if thisSPS.nSP > 0:
            #if CalPref or thissett.saddle_point["CalBarrsInData"]: thisSPS.get_SP_type(SPlist=thisSPS.SPlist)
            thisDataSPs = postSPS.insert_AVSP2DataSPs(thisDataSPs, thisSPS, idav, DFWriter, Precursor=Precursor)
            if Precursor and thissett.defect_bank["Recycle"]:
                DefectBank_list = postSPS.add_to_DefectBank(thissett, thisAV, thisSPS, isRecycled, isPGSYMM,
                                                            thisSOPs.sch_symbol, DefectBank_list, out_paths[3])
            if Precursor:
                logstr = f"Found {str(thisSPS.nSP)} saddle points in {str(idav)} precursor volume!"
            else:
                logstr = f"Found {str(thisSPS.nSP)} saddle points in {str(idav)} active volume!"
            logstr += "\n" + "-----------------------------------------------------------------"
            LogWriter.write_data(logstr)

        if not Precursor: Dataout.visualize_AV_SPs(thissett.visual, seakmcdata, thisAVitags, thisAV, thisSPS, istep,
                                                   idav, out_paths[0])

    iav += 1
    thisfinished_AVs += 1
    thisundo_idavs = np.delete(thisundo_idavs, np.argwhere(thisundo_idavs == idav))

    tocav = time.time()
    if Precursor:
        logstr = f"Total time for {idav} precursor volume: {round(tocav - ticav, float_precision)} s"
    else:
        logstr = f"Total time for {idav} active volume: {round(tocav - ticav, float_precision)} s"
    logstr += "\n" + "-----------------------------------------------------------------"
    LogWriter.write_data(logstr)

    if not Precursor:
        if thissett.system["Restart"]["WriteRestart"] and thisfinished_AVs % thissett.system["Restart"][
            "AVStep4Restart"] == 0:
            thisRestart = RESTART(istep, thisfinished_AVs, DefectBank_list, thisSuperBasin, seakmcdata, Eground,
                                  thisDataSPs, thisAVitags, df_delete_SPs, thisundo_idavs, simulation_time)
            thisRestart.to_file()
            thisRestart = None

    return seakmcdata, thisDataSPs, thisAVitags


def data_find_saddlepoints(istep, thissett, seakmcdata, DefectBank_list, thisSuperBasin, Eground,
                           DataSPs, AVitags, df_delete_SPs, undo_idavs, finished_AVs, simulation_time, object_dict):
    LogWriter = object_dict['LogWriter']

    iav = 0
    thiscolor = 0
    preSPS.initialization_thisdata(seakmcdata, thissett)
    if thissett.spsearch["insituGuidedSPS"] and seakmcdata.nprecursors > 0:
        logstr = f"There are total {seakmcdata.nprecursors} precursor volumes."
        logstr += "\n" + "-----------------------------------------------------------------"
        LogWriter.write_data(logstr)

    for idav in undo_idavs:
        thisAV = preSPS.initialize_thisAV(seakmcdata, idav, Rebuild=False, Precursor=False)

        if thisAV is not None:
            AVitags[idav] = thisAV.itags[0:(thisAV.nactive + thisAV.nbuffer)]
            if thissett.spsearch["insituGuidedSPS"] and seakmcdata.nprecursors > 0:
                emptya = np.array([], dtype=int)
                thisAVitags = [emptya for i in range(seakmcdata.nprecursors)]
                thisDataSPs = Data_SPs(istep, seakmcdata.nprecursors)
                thisDataSPs.initialization()

                thisundo_idavs = seakmcdata.def_atoms[idav]["itag"]
                thisfinished_AVs = 0
                localiav = 0

                logstr = f"The precursor IDs: {thisundo_idavs} AV ID:{idav}."
                logstr += "\n" + "-----------------------------------------------------------------"
                LogWriter.write_data(logstr)
                for idpre in thisundo_idavs:
                    thisprecursor = preSPS.initialize_thisAV(seakmcdata, idpre, Rebuild=False, Precursor=True)
                    thisAVitags[idpre] = thisprecursor.itags[0:(thisprecursor.nactive + thisprecursor.nbuffer)]

                    local_coords, thisVNS = preSPS.initialize_AV_props(thisprecursor)
                    thisSOPs, isPGSYMM = preSPS.get_SymmOperators(thissett, thisprecursor, idpre,
                                                                  PointGroup=thissett.active_volume["PointGroupSymm"])
                    #thisStrain = preSPS.get_AV_atom_strain(thisprecursor, thissett, thiscolor)
                    isRecycled, Pre_Disps = preSPS.get_Pre_Disps(idpre, thisprecursor, thissett, thisSOPs,
                                                                 DefectBank_list, istep)
                    thisnspsearch = thissett.spsearch['NSearch']
                    SNC, CalPref, errorlog = preSPS.initial_SNC_CalPref(idpre, thisprecursor, thissett, Precursor=True,
                                                                        insituGSPS=False)
                    if len(errorlog) > 0: LogWriter.write_data(errorlog)

                    dynmatAV = None
                    if SNC or CalPref:
                        SNC, CalPref, dynmatAV = preSPS.get_thisSNC4spsearch(idpre, thissett, thisprecursor,
                                                                             SNC, CalPref, object_dict, thiscolor, istep)

                    seakmcdata, thisDataSPs, thisAVitags = data_find_saddlepointsearch(localiav, idpre, thisprecursor,
                                                                                       local_coords, thisSOPs, isPGSYMM,
                                                                                       thisVNS, isRecycled, Pre_Disps,
                                                                                       thisnspsearch, SNC, CalPref,
                                                                                       dynmatAV,
                                                                                       istep, thissett, seakmcdata,
                                                                                       DefectBank_list, thisSuperBasin,
                                                                                       Eground,
                                                                                       thisDataSPs, thisAVitags,
                                                                                       df_delete_SPs, thisundo_idavs,
                                                                                       thisfinished_AVs,
                                                                                       simulation_time,
                                                                                       object_dict, Precursor=True,
                                                                                       insituGSPS=False)

                local_coords, thisVNS = preSPS.initialize_AV_props(thisAV)
                thisSOPs, isPGSYMM = preSPS.get_SymmOperators(thissett, thisAV, idav, PointGroup=False)
                #thisStrain = preSPS.get_AV_atom_strain(thisAV, thissett, thiscolor)
                isRecycled, Pre_Disps, precursorbarrs = preSPS.get_Pre_Disps_from_precursor(idav, thisAV, thissett,
                                                                                            seakmcdata, AVitags[idav],
                                                                                            thisDataSPs, thisAVitags)
                thisnsps = len(Pre_Disps)
                if isinstance(thissett.spsearch["NMaxSPs4insituGSPS"], int):
                    thisNMaxSPs = max(int(thissett.spsearch["NMaxSPs4insituGSPS"] / len(undo_idavs)), 1)
                    if len(Pre_Disps) > thisNMaxSPs:
                        ai = np.argsort(precursorbarrs)
                        ai = ai[0:thisNMaxSPs]
                        Pre_Disps = list(map(lambda i: Pre_Disps[i], ai))
                        ai = None
                else:
                    pass
                thisnspsearch = len(Pre_Disps)
                precursorbarrs = []
                thisAVitags = []
                thisDataSPs = None

                logstr = "\n" + f"Finished SPS on precursor volumes of {idav} AV and found {thisnsps} saddle points."
                logstr += "\n" + f"The first {thisnspsearch} lowest barrier saddle points are subjected to insituGSPS."
                logstr += "\n" + "-----------------------------------------------------------------"
                LogWriter.write_data(logstr)

                SNC, CalPref, errorlog = preSPS.initial_SNC_CalPref(idav, thisAV, thissett, Precursor=False,
                                                                    insituGSPS=True)
                if len(errorlog) > 0: LogWriter.write_data(errorlog)
                dynmatAV = None
                if SNC or CalPref:
                    SNC, CalPref, dynmatAV = preSPS.get_thisSNC4spsearch(idav, thissett, thisAV,
                                                                         SNC, CalPref, object_dict, thiscolor, istep)

                seakmcdata, DataSPs, AVitags = data_find_saddlepointsearch(iav, idav, thisAV,
                                                                           local_coords, thisSOPs, isPGSYMM, thisVNS,
                                                                           isRecycled, Pre_Disps,
                                                                           thisnspsearch, SNC, CalPref, dynmatAV,
                                                                           istep, thissett, seakmcdata, DefectBank_list,
                                                                           thisSuperBasin, Eground,
                                                                           DataSPs, AVitags, df_delete_SPs, undo_idavs,
                                                                           finished_AVs, simulation_time,
                                                                           object_dict, Precursor=False,
                                                                           insituGSPS=True)
                Pre_Disps = []

            else:
                local_coords, thisVNS = preSPS.initialize_AV_props(thisAV)
                thisSOPs, isPGSYMM = preSPS.get_SymmOperators(thissett, thisAV, idav,
                                                              PointGroup=thissett.active_volume["PointGroupSymm"])
                #thisStrain = preSPS.get_AV_atom_strain(thisAV, thissett, thiscolor)
                isRecycled, Pre_Disps = preSPS.get_Pre_Disps(idav, thisAV, thissett, thisSOPs, DefectBank_list, istep)
                thisnspsearch = thissett.spsearch["NSearch"]
                SNC, CalPref, errorlog = preSPS.initial_SNC_CalPref(idav, thisAV, thissett, Precursor=False,
                                                                    insituGSPS=False)
                if len(errorlog) > 0: LogWriter.write_data(errorlog)

                dynmatAV = None
                if SNC or CalPref:
                    SNC, CalPref, dynmatAV = preSPS.get_thisSNC4spsearch(idav, thissett, thisAV,
                                                                         SNC, CalPref, object_dict, thiscolor, istep)

                seakmcdata, DataSPs, AVitags = data_find_saddlepointsearch(iav, idav, thisAV,
                                                                           local_coords, thisSOPs, isPGSYMM, thisVNS,
                                                                           isRecycled, Pre_Disps,
                                                                           thisnspsearch, SNC, CalPref, dynmatAV,
                                                                           istep, thissett, seakmcdata, DefectBank_list,
                                                                           thisSuperBasin, Eground,
                                                                           DataSPs, AVitags, df_delete_SPs, undo_idavs,
                                                                           finished_AVs, simulation_time,
                                                                           object_dict, Precursor=False,
                                                                           insituGSPS=False)

    thisAV = None
    thisSPS = None
    thisSOPs = None
    dynmatAV = None
    return seakmcdata, DataSPs, AVitags
