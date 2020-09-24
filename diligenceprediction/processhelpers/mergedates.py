import pandas as pd
import numpy as np

from .campdatehelpers import getANMCampIds, afterSystemOnline, getAllCampIds, arrangeDate


class MergeDates:
    """
      Class to handle merging of camp (recorded) dates,
      since the data recorded day might not represent the actual camp date.

    """

    def __init__(self, online_day, threshold=3, searchRange=7, expandClusterTimes=2):
        """
           sets merge parameters

           Parameters
           ----------
           threshold : int
                threshold for the dates to be clustered if number of patients is less than or equal to
           searchRange : int
                number of days the searching is done
           expandClusterTimes : int
                how may times the cluster is expanded in scenario 1
           online_day : datetime object
                dataframe starting day

        """

        self.threshold = threshold
        self.searchRange = searchRange
        self.expandClusterTimes = expandClusterTimes
        self.online_day = online_day

    def mergeAlgo(self, df):
        """
          runs the merge algo for all camps in all ANMs and return the date clusters with the merged dates for each ANM

          Step 1. finds clusters for values less than the treshold by searching the given range
          Step 2. expands the cluster given number of times by searching given range into past
          Step 3. handle if there is a value larger than threshold closer to considered date than to the date with highest value in the cluster

          Parameters
          ----------
          df : pandas DataFrame
              Dataframe to be processed (camp-datewise preprocessed input)

          Returns
          -------
          dict_campClusterDates : dictionary
              a dictionary with dates to be clustered together for each camp_id
          dict_campUnifiedDate : dictionary
              a dictionary with the cluster dates to be represented after merging few dates together for each camp_id
          """

        # get all ANM and camps for clustering
        testANM, testcamp = getANMCampIds(df)

        dict_campClusterDates = {}
        dict_campUnifiedDate = {}

        originalDatesDF = self.getOriginalGroupedByDate(df)
        originalDatesDF2 = afterSystemOnline(originalDatesDF, self.online_day)

        sortedOriginalDates = originalDatesDF2.sort_values(by=['date'], inplace=False, ascending=False)
        print("Date range: ", sortedOriginalDates['date'].iloc[0], sortedOriginalDates['date'].iloc[-1])

        # dummy dataframe
        dates = pd.date_range(start=sortedOriginalDates['date'].iloc[-1], end=sortedOriginalDates['date'].iloc[0],
                              freq='D')
        dummydates = pd.DataFrame({'date': dates, 'Val': np.random.randn(len(dates))})

        for i in range(len(testANM)):
            for campid in testcamp[i]:
                # print("CAMP: ", campid)
                campClusterDates = []
                campUnifiedDate = []
                campdf = sortedOriginalDates[(sortedOriginalDates['sub_center_id'] == testANM[i]) & (
                        sortedOriginalDates['camp_id'] == campid)].reset_index()

                # merge with dummy dataframe
                df_merge = pd.merge(dummydates, campdf, on='date', how='left')
                df_merge['Num_patients'].fillna(0, inplace=True)

                datesOriginal = df_merge['date']
                patientsOriginal = df_merge['Num_patients']
                Num_dates = len(df_merge['date'])
                datesMerged = np.array(df_merge['date'], dtype=object)
                patientsMerged = np.array(df_merge['Num_patients'])

                # finds clusters for values less than the treshold
                for d in range(0, Num_dates):
                    if ((patientsMerged[d] <= self.threshold) and (patientsMerged[d] > 0)):
                        clusterDates = [datesMerged[d]]
                        clusterPatients = [patientsMerged[d]]
                        clusterIdx = [d]

                        # search for non-zero values in +/- searchRange (end value inclusive)
                        for r in range(1, self.searchRange + 1):
                            # search searchRange days to future
                            if ((d - r) > 0):
                                if (patientsMerged[d - r] > 0):
                                    clusterDates.append(datesMerged[d - r])
                                    clusterPatients.append(patientsMerged[d - r])
                                    clusterIdx.append(d - r)

                            # search searchRange days to the past
                            if ((d + r) < Num_dates):
                                if (patientsMerged[d + r] > 0):
                                    clusterDates.append(datesMerged[d + r])
                                    clusterPatients.append(patientsMerged[d + r])
                                    clusterIdx.append(d + r)

                        # scenario 1
                        clusterEarliestDayIdx = np.max(clusterIdx)
                        if clusterEarliestDayIdx > d:
                            flag = False
                            # expands the cluster given number of times by searching given range into past
                            for t in range(self.expandClusterTimes):
                                # search searchRange days to the past
                                for r in range(1, self.searchRange + 1):
                                    if ((clusterEarliestDayIdx + r) < Num_dates):
                                        if (patientsMerged[clusterEarliestDayIdx + r] > self.threshold):
                                            # stops cluster expansion if a value greater than threshold is found
                                            flag = True
                                            break
                                        elif (patientsMerged[clusterEarliestDayIdx + r] > 0):
                                            # expands cluster if a non-zero value less than threshold is found
                                            clusterDates.append(datesMerged[clusterEarliestDayIdx + r])
                                            clusterPatients.append(patientsMerged[clusterEarliestDayIdx + r])
                                            clusterIdx.append(clusterEarliestDayIdx + r)
                                if flag:
                                    break
                                clusterEarliestDayIdx = np.max(clusterIdx)

                        # scenario 2
                        clusterPatients = np.array(clusterPatients)
                        clusterDates = np.array(clusterDates)
                        clusterIdx = np.array(clusterIdx)
                        belowThresholdIdx = clusterIdx[clusterPatients <= self.threshold]
                        # select the leftmost below threshold value idx
                        pastBelowThresholdIdx = np.max(belowThresholdIdx)
                        max_idx = clusterIdx[np.argmax(clusterPatients)]
                        if (max_idx < pastBelowThresholdIdx):
                            # search searchRange-1 days backwards for values above threshold
                            for r in range(1, self.searchRange):
                                if ((pastBelowThresholdIdx + r) < Num_dates):
                                    if (patientsMerged[pastBelowThresholdIdx + r] > self.threshold):
                                        if (r >= (pastBelowThresholdIdx - max_idx)):
                                            break
                                        else:
                                            # if there is a value larger than threshold closer to considered date than
                                            # to the date with highest value in the cluster
                                            # remove all leftmost values from considered date
                                            remove_val = pastBelowThresholdIdx + r
                                            clusterIdx = clusterIdx[np.where(clusterIdx < remove_val)]
                                            clusterPatients = clusterPatients[np.where(clusterIdx < remove_val)]
                                            clusterDates = clusterDates[np.where(clusterIdx < remove_val)]

                        if len(clusterPatients) == 1:
                            # no non zero values found in searchRange
                            continue
                        else:
                            # if cluster is found, merge them together and assign to the date with highest patients
                            max_idx = clusterIdx[np.argmax(clusterPatients)]
                            clusterSum = np.sum(clusterPatients)

                            for c in clusterIdx:
                                if c == max_idx:
                                    patientsMerged[c] = clusterSum
                                else:
                                    patientsMerged[c] = 0

                        campClusterDates.append(clusterDates)
                        campUnifiedDate.append(datesMerged[max_idx])
                        # print(clusterDates)

                dict_campClusterDates[campid] = campClusterDates
                dict_campUnifiedDate[campid] = campUnifiedDate

        return dict_campClusterDates, dict_campUnifiedDate

    def addClusters(self, dfm, df):
        """
          Gets the cluster dates and merged dates for each camp by running the merge algo
          and label all the records accordingly with new merged dates

          Parameters
          ----------
          dfm : pandas DataFrame
              Dataframe to be processed (camp-datewise preprocessed input)
          df : pandas DataFrame
              Dataframe to be processed (original)

          Returns
          -------
          dfm : pandas DataFrame
              Dataframe with the added cluster dates
          """

        dict_campClusterDates, dict_campUnifiedDate = self.mergeAlgo(df)

        dfm['cluster_date'] = np.nan
        campIds = getAllCampIds(df)

        for i in range(len(campIds)):
            campClusterDates = dict_campClusterDates[campIds[i]]
            campUnifiedDate = dict_campUnifiedDate[campIds[i]]

            for j in range(len(campUnifiedDate)):
                for k in range(len(campClusterDates[j])):
                    mask = (dfm['date'] == campClusterDates[j][k]) & (dfm['camp_id'] == campIds[i])
                    dfm.loc[mask, 'cluster_date'] = campUnifiedDate[j]

        dfm['cluster_date'] = pd.to_datetime(dfm['cluster_date'])
        dfm['cluster_date'].fillna(dfm['date'], inplace=True)

        return dfm

    def getOriginalGroupedByDate(self, df):
        columns = ['sub_center_id', 'camp_id', 'ANC_Mother Id', 'ANC_ANC1 Date', 'ANC_ANC2 Date', 'ANC_ANC3 Date',
                   'ANC_ANC4 Date']
        dfmod = df.loc[df['ANC_Mother Id'].notnull(), columns]
        newcolumns = ['date']
        oldcolumns = [['ANC_ANC1 Date', 'ANC_ANC2 Date', 'ANC_ANC3 Date', 'ANC_ANC4 Date']]

        df2 = arrangeDate(dfmod, newcolumns, oldcolumns)

        grouped = df2.groupby(['sub_center_id', 'camp_id', 'date'])
        df3 = grouped.agg(
            Num_patients=pd.NamedAgg(column='ANC_Mother Id', aggfunc=lambda x: len(tuple(x)))
        ).reset_index()

        return df3
