import pandas as pd
import numpy as np

from .kdes import KDEs


class ShortRangeRules:
    """
      This class processes the short range rules and outputs the non-diligence probability vectors
      of ANMs in a timeframe using KDE.

      Short range rules are the rules that are calculated at health camp level.

      At initialization, the KDEs are obtained using R scripts. Percentages are obtained after processing each rule
      and they are converted to non-diligence probability using KDE, by taking the area under the curve from 0 to x or
      x to 100 according to the nature of the rule.

      Aggregate over one month: the health camps have lot of variability so instead of looking at one health camp
      we consider the average of probability of non-diligence probability over all health camps in 4 weeks.
      This is the short-term rule fraud probability.

      """

    def __init__(self, kdes):

        self.func_get_prob_mass_trans = kdes.func_get_prob_mass_trans
        self.bpsd_fit = kdes.bpsd_fit
        self.urine_fit = kdes.urine_fit
        self.hb50_fit = kdes.hb50_fit
        self.hb70_fit = kdes.hb70_fit
        self.hb7_fit = kdes.hb7_fit
        self.bphyper_fit = kdes.bphyper_fit
        self.bp_contra_fit = kdes.bp_contra_fit
        self.weight_contra_fit = kdes.weight_contra_fit
        self.hb_contra_fit = kdes.hb_contra_fit
        self.bsugar_contra_fit = kdes.bsugar_contra_fit
        self.fetalhr_contra_fit = kdes.fetalhr_contra_fit
        self.usugar_contra_fit = kdes.usugar_contra_fit
        self.alb_contra_fit = kdes.alb_contra_fit

    def getBPsdRuleData(self, df, dummydf):

        """
          Rule: BP values: If majority (>70%) of measurements are 120/80 or 110/70

          Step 1. Processes the data in the given timeframe using the rule
          Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
          Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

          Parameters
          ----------
          df : pandas DataFrame
              Dataframe to be processed
          dummydf : pandas DataFrame
              Dataframe with all the ANMs

          Returns
          -------
          list
              a list of non-diligence probabilities of ANMs
          """

        # Step 1
        grouped = df.groupby(['sub_center_id', 'camp_id', 'cluster_date'])
        bp = grouped.agg({'ANC_Mother Id': lambda x: len(list(x)),
                          'BPsys': lambda x: list(x),
                          'BPdia': lambda x: list(x)}).reset_index()

        # Step 2
        badcount = []
        totalcount = []
        for k in range(len(bp)):
            c = 0
            tot = 0
            for i in range(len(bp.loc[k, 'BPsys'])):
                if (bp.loc[k, 'BPsys'][i] == 120) and (bp.loc[k, 'BPdia'][i] == 80):
                    c += 1
                    tot += 1
                elif (bp.loc[k, 'BPsys'][i] == 110) and (bp.loc[k, 'BPdia'][i] == 70):
                    c += 1
                    tot += 1
                elif (bp.loc[k, 'BPsys'][i] > 7) or (bp.loc[k, 'BPdia'][i] > 7):
                    tot += 1
            badcount.append(c)
            totalcount.append(tot)

        badbpcount = pd.Series(badcount)
        bp = bp.assign(badbpcount=badbpcount.values)
        totalbpcount = pd.Series(totalcount)
        bp = bp.assign(totalbpcount=totalbpcount.values)
        bp['totalbpcount'].fillna(0, inplace=True)
        bp['suspicious_percentage'] = bp.apply(
            lambda row: row['badbpcount'] / row['totalbpcount'] * 100 if row['totalbpcount'] != 0 else 0, axis=1)

        anmgrouped = bp.groupby(['sub_center_id'])
        bpanm = anmgrouped.agg(
            Num_camps=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: len(list(x))),
            Tot_num_patients=pd.NamedAgg(column='ANC_Mother Id', aggfunc=sum),
            suspicious_percentage=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: list(x)),
            cluster_dates=pd.NamedAgg(column='cluster_date', aggfunc=lambda x: list(x))
        ).reset_index()

        # Step 3
        fraud_probabilities = []
        for i in range(len(bpanm)):
            fraud_prob = []
            for j in range(len(bpanm.loc[i, 'suspicious_percentage'])):
                x = bpanm.loc[i, 'suspicious_percentage'][j]
                prob = self.func_get_prob_mass_trans(self.bpsd_fit, 0, x)
                fraud_prob.append(prob)
            fraud_probabilities.append(np.mean(fraud_prob))

        fraud_probabilities = pd.Series(fraud_probabilities)
        bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe with all ANM
        df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def getModfUrineRule(self, df, dummydf):
        """
          RULE: Urine test: greater than 90% of values marked as “Absent”; more than 10% should be Present or Test Not Done

          Step 1. Processes the data in the given timeframe using the rule
          Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
          Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

          Parameters
          ----------
          df : pandas DataFrame
              Dataframe to be processed
          dummydf : pandas DataFrame
              Dataframe with all the ANMs

          Returns
          -------
          list
              a list of non-diligence probabilities of ANMs
          """
        # Step 1
        grouped = df.groupby(['sub_center_id', 'camp_id', 'cluster_date'])
        ur = grouped.agg({'ANC_Mother Id': lambda x: len(list(x)),
                          'urinesugar': lambda x: list(x),
                          'albumin': lambda x: list(x)}).reset_index()

        # Step 2
        absentcount = []
        totcount = []
        for k in range(len(ur)):
            abcountanm = 0
            totcountanm = 0
            for i in range(len(ur.loc[k, 'albumin'])):
                if (ur.loc[k, 'albumin'][i] == 'False') and (ur.loc[k, 'urinesugar'][i] == 'False'):
                    abcountanm += 1
                    totcountanm += 1
                elif (ur.loc[k, 'albumin'][i] == 'False') and (ur.loc[k, 'urinesugar'][i] == 'True'):
                    totcountanm += 1
                elif (ur.loc[k, 'albumin'][i] == 'True') and (ur.loc[k, 'urinesugar'][i] == 'False'):
                    totcountanm += 1
                elif (ur.loc[k, 'albumin'][i] == 'True') and (ur.loc[k, 'urinesugar'][i] == 'True'):
                    totcountanm += 1
            absentcount.append(abcountanm)
            totcount.append(totcountanm)

        absenturcount = pd.Series(absentcount)
        ur = ur.assign(absenturcount=absenturcount.values)
        toturcount = pd.Series(totcount)
        ur = ur.assign(toturcount=toturcount.values)
        ur['falsepercentage'] = ur.apply(
            lambda row: row['absenturcount'] / row['toturcount'] * 100 if row['toturcount'] != 0 else 0, axis=1)

        anmgrouped = ur.groupby(['sub_center_id'])
        uranm = anmgrouped.agg(
            Num_camps=pd.NamedAgg(column='falsepercentage', aggfunc=lambda x: len(list(x))),
            Tot_num_patients=pd.NamedAgg(column='ANC_Mother Id', aggfunc=sum),
            false_percentage=pd.NamedAgg(column='falsepercentage', aggfunc=lambda x: list(x)),
            cluster_dates=pd.NamedAgg(column='cluster_date', aggfunc=lambda x: list(x))
        ).reset_index()

        # Step 3
        fraud_probabilities = []
        for i in range(len(uranm)):
            fraud_prob = []
            for j in range(len(uranm.loc[i, 'false_percentage'])):
                x = uranm.loc[i, 'false_percentage'][j]
                prob = self.func_get_prob_mass_trans(self.urine_fit, 0, x)
                fraud_prob.append(prob)
            fraud_probabilities.append(np.mean(fraud_prob))

        fraud_probabilities = pd.Series(fraud_probabilities)
        uranm = uranm.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, uranm, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def getBPhyperRule(self, df, dummydf):
        """
          "RULE: Proportion of patients with Hypertension (either BP Sys > 140 OR BP Dia > 90) < 5%"

          Step 1. Processes the data in the given timeframe using the rule
          Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
          Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

          Parameters
          ----------
          df : pandas DataFrame
              Dataframe to be processed
          dummydf : pandas DataFrame
              Dataframe with all the ANMs

          Returns
          -------
          list
              a list of non-diligence probabilities of ANMs
          """
        # Step 1
        grouped = df.groupby(['sub_center_id', 'camp_id', 'cluster_date'])
        bp = grouped.agg({'ANC_Mother Id': lambda x: len(list(x)),
                          'BPsys': lambda x: list(x),
                          'BPdia': lambda x: list(x)}).reset_index()

        # Step 2
        badcount = []
        totalcount = []
        for k in range(len(bp)):
            c = 0
            tot = 0
            for i in range(len(bp.loc[k, 'BPsys'])):
                if (bp.loc[k, 'BPsys'][i] > 140) or (bp.loc[k, 'BPdia'][i] > 90):
                    c += 1
                    tot += 1
                elif (bp.loc[k, 'BPsys'][i] > 7) or (bp.loc[k, 'BPdia'][i] > 7):
                    tot += 1
            badcount.append(c)
            totalcount.append(tot)

        badbpcount = pd.Series(badcount)
        bp = bp.assign(badbpcount=badbpcount.values)
        totalbpcount = pd.Series(totalcount)
        bp = bp.assign(totalbpcount=totalbpcount.values)
        bp['totalbpcount'].fillna(0, inplace=True)
        bp['hyper_percentage'] = bp.apply(
            lambda row: row['badbpcount'] / row['totalbpcount'] * 100 if row['totalbpcount'] != 0 else 0, axis=1)

        anmgrouped = bp.groupby(['sub_center_id'])
        bpanm = anmgrouped.agg(
            Num_camps=pd.NamedAgg(column='hyper_percentage', aggfunc=lambda x: len(list(x))),
            Tot_num_patients=pd.NamedAgg(column='ANC_Mother Id', aggfunc=sum),
            hyper_percentage=pd.NamedAgg(column='hyper_percentage', aggfunc=lambda x: list(x)),
            dates=pd.NamedAgg(column='cluster_date', aggfunc=lambda x: list(x))
        ).reset_index()

        # Step 3
        fraud_probabilities = []
        for i in range(len(bpanm)):
            fraud_prob = []
            for j in range(len(bpanm.loc[i, 'hyper_percentage'])):
                x = bpanm.loc[i, 'hyper_percentage'][j]
                prob = self.func_get_prob_mass_trans(self.bphyper_fit, x, 100)
                fraud_prob.append(prob)
            fraud_probabilities.append(np.mean(fraud_prob))

        fraud_probabilities = pd.Series(fraud_probabilities)
        bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def getHBrule(self, df, dummydf):
        """
          "RULE: Proportion of Patients with Anemia (defined as HB < 11) < 48% or >68%"

          Step 1. Processes the data in the given timeframe using the rule
          Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
          Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

          Parameters
          ----------
          df : pandas DataFrame
              Dataframe to be processed
          dummydf : pandas DataFrame
              Dataframe with all the ANMs

          Returns
          -------
          list
              a list of non-diligence probabilities of ANMs
          """
        # Step 1
        grouped = df.groupby(['sub_center_id', 'camp_id', 'cluster_date'])
        bp = grouped.agg({'ANC_Mother Id': lambda x: len(list(x)),
                          'hb': lambda x: list(x)}).reset_index()

        # Step 2
        anaemiaCount = []
        totalRealCount = []
        for k in range(len(bp)):
            anaemia = 0
            totReal = 0
            for i in range(len(bp.loc[k, 'hb'])):
                if (bp.loc[k, 'hb'][i] < 11) and (bp.loc[k, 'hb'][i] > 5):
                    anaemia += 1
                    totReal += 1
                elif (bp.loc[k, 'hb'][i] > 5):
                    totReal += 1

            anaemiaCount.append(anaemia)
            totalRealCount.append(totReal)

        bp = bp.assign(anaemiaCount=anaemiaCount)
        bp = bp.assign(totalRealCount=totalRealCount)
        bp['totalRealCount'].fillna(0, inplace=True)
        bp['anaemia_percentage'] = bp.apply(
            lambda row: row['anaemiaCount'] / row['totalRealCount'] * 100 if row['totalRealCount'] != 0 else 0, axis=1)

        anmgrouped = bp.groupby(['sub_center_id'])
        bpanm = anmgrouped.agg(
            Num_camps=pd.NamedAgg(column='anaemia_percentage', aggfunc=lambda x: len(list(x))),
            Tot_num_patients=pd.NamedAgg(column='ANC_Mother Id', aggfunc=sum),
            anaemia_percentage=pd.NamedAgg(column='anaemia_percentage', aggfunc=lambda x: list(x)),
            dates=pd.NamedAgg(column='cluster_date', aggfunc=lambda x: list(x))
        ).reset_index()

        # Step 3
        fraud_probabilities = []
        for i in range(len(bpanm)):
            fraud_prob = []
            for j in range(len(bpanm.loc[i, 'anaemia_percentage'])):
                x = bpanm.loc[i, 'anaemia_percentage'][j]
                if (x < 50):
                    probR = self.func_get_prob_mass_trans(self.hb50_fit, x, 50)
                    prob = probR[0]
                elif (x > 70):
                    probR = self.func_get_prob_mass_trans(self.hb70_fit, 70, x)
                    prob = probR[0]
                else:
                    prob = 0.0

                fraud_prob.append(prob)
            fraud_probabilities.append(np.mean(fraud_prob))

        fraud_probabilities = pd.Series(fraud_probabilities)
        bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def getHB7rule(self, df, dummydf):
        """
          "RULE: Proportion of Patients with HB < 7 < 5%"

          Step 1. Processes the data in the given timeframe using the rule
          Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
          Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

          Parameters
          ----------
          df : pandas DataFrame
              Dataframe to be processed
          dummydf : pandas DataFrame
              Dataframe with all the ANMs

          Returns
          -------
          list
              a list of non-diligence probabilities of ANMs
          """
        # Step 1
        grouped = df.groupby(['sub_center_id', 'camp_id', 'cluster_date'])
        bp = grouped.agg({'ANC_Mother Id': lambda x: len(list(x)),
                          'hb': lambda x: list(x)}).reset_index()

        # Step 2
        anaemiaCount = []
        totalRealCount = []
        for k in range(len(bp)):
            anaemia = 0
            totReal = 0
            for i in range(len(bp.loc[k, 'hb'])):
                if (bp.loc[k, 'hb'][i] < 7) and (bp.loc[k, 'hb'][i] > 5):
                    anaemia += 1
                    totReal += 1
                elif (bp.loc[k, 'hb'][i] > 5):
                    totReal += 1

            anaemiaCount.append(anaemia)
            totalRealCount.append(totReal)

        bp = bp.assign(anaemiaCount=anaemiaCount)
        bp = bp.assign(totalRealCount=totalRealCount)
        bp['totalRealCount'].fillna(0, inplace=True)
        bp['anaemia_percentage'] = bp.apply(
            lambda row: row['anaemiaCount'] / row['totalRealCount'] * 100 if row['totalRealCount'] != 0 else 0, axis=1)

        anmgrouped = bp.groupby(['sub_center_id'])
        bpanm = anmgrouped.agg(
            Num_camps=pd.NamedAgg(column='anaemia_percentage', aggfunc=lambda x: len(list(x))),
            Tot_num_patients=pd.NamedAgg(column='ANC_Mother Id', aggfunc=sum),
            anaemia_percentage=pd.NamedAgg(column='anaemia_percentage', aggfunc=lambda x: list(x)),
            dates=pd.NamedAgg(column='cluster_date', aggfunc=lambda x: list(x))
        ).reset_index()

        # Step 3
        fraud_probabilities = []
        for i in range(len(bpanm)):
            fraud_prob = []
            for j in range(len(bpanm.loc[i, 'anaemia_percentage'])):
                x = bpanm.loc[i, 'anaemia_percentage'][j]
                prob = self.func_get_prob_mass_trans(self.hb7_fit, x, 100)
                fraud_prob.append(prob)
            fraud_probabilities.append(np.mean(fraud_prob))

        fraud_probabilities = pd.Series(fraud_probabilities)
        bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def bpcontradictionRule(self, df, dummydf):
        """
          "RULE: “No-Equipment/Entered-Data Contradiction” for blood pressure (No Equipment and then “Present/Absent for another patient)"

          Step 1. Processes the data in the given timeframe using the rule
          Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
          Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

          Parameters
          ----------
          df : pandas DataFrame
              Dataframe to be processed
          dummydf : pandas DataFrame
              Dataframe with all the ANMs

          Returns
          -------
          list
              a list of non-diligence probabilities of ANMs
          """
        # Step 1
        grouped = df.groupby(['sub_center_id', 'camp_id', 'cluster_date'])
        bp = grouped.agg(
            Num_patients=pd.NamedAgg(column='ANC_Mother Id', aggfunc=lambda x: len(list(x))),
            Num_checked_patients=pd.NamedAgg(column='BPsys', aggfunc=lambda x: len(list(x.dropna()))),
            BPsys=pd.NamedAgg(column='BPsys', aggfunc=lambda x: list(x.dropna())),
            BPdia=pd.NamedAgg(column='BPdia', aggfunc=lambda x: list(x.dropna()))
        ).reset_index()

        # Step 2
        noEquipCount = []
        totalRealCount = []
        for k in range(len(bp)):
            noEquip = 0
            totReal = 0
            for i in range(len(bp.loc[k, 'BPsys'])):
                if (bp.loc[k, 'BPsys'][i] == 4):
                    noEquip += 1
                elif (bp.loc[k, 'BPsys'][i] > 7):
                    totReal += 1

            noEquipCount.append(noEquip)
            totalRealCount.append(totReal)

        bp = bp.assign(noEquipCount=noEquipCount)
        bp = bp.assign(totalRealCount=totalRealCount)
        bp['totalRealCount'].fillna(0, inplace=True)
        bp['suspicious_percentage'] = bp.apply(
            lambda row: row['noEquipCount'] / row['Num_checked_patients'] * 100 if row['totalRealCount'] != 0 else 0,
            axis=1)

        anmgrouped = bp.groupby(['sub_center_id'])
        bpanm = anmgrouped.agg(
            Num_camps=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: len(list(x))),
            Tot_num_patients=pd.NamedAgg(column='Num_patients', aggfunc=sum),
            suspicious_percentage=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: list(x)),
            dates=pd.NamedAgg(column='cluster_date', aggfunc=lambda x: list(x))
        ).reset_index()

        # Step 3
        fraud_probabilities = []
        for i in range(len(bpanm)):
            fraud_prob = []
            for j in range(len(bpanm.loc[i, 'suspicious_percentage'])):
                x = bpanm.loc[i, 'suspicious_percentage'][j]
                prob = self.func_get_prob_mass_trans(self.bp_contra_fit, 0, x)
                fraud_prob.append(prob)
            fraud_probabilities.append(np.mean(fraud_prob))

        fraud_probabilities = pd.Series(fraud_probabilities)
        bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def weightcontradictionRule(self, df, dummydf):
        """
          "RULE: “No-Equipment/Entered-Data Contradiction” for weight (No Equipment and then “Present/Absent for another patient)"

          Step 1. Processes the data in the given timeframe using the rule
          Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
          Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

          Parameters
          ----------
          df : pandas DataFrame
              Dataframe to be processed
          dummydf : pandas DataFrame
              Dataframe with all the ANMs

          Returns
          -------
          list
              a list of non-diligence probabilities of ANMs
          """
        # Step 1
        grouped = df.groupby(['sub_center_id', 'camp_id', 'cluster_date'])
        bp = grouped.agg(
            Num_patients=pd.NamedAgg(column='ANC_Mother Id', aggfunc=lambda x: len(list(x))),
            Num_checked_patients=pd.NamedAgg(column='weight', aggfunc=lambda x: len(list(x.dropna()))),
            weight=pd.NamedAgg(column='weight', aggfunc=lambda x: list(x.dropna()))
        ).reset_index()

        # Step 2
        noEquipCount = []
        totalRealCount = []
        for k in range(len(bp)):
            noEquip = 0
            totReal = 0
            for i in range(len(bp.loc[k, 'weight'])):
                if (bp.loc[k, 'weight'][i] == 4):
                    noEquip += 1
                elif (bp.loc[k, 'weight'][i] > 7):
                    totReal += 1

            noEquipCount.append(noEquip)
            totalRealCount.append(totReal)

        bp = bp.assign(noEquipCount=noEquipCount)
        bp = bp.assign(totalRealCount=totalRealCount)
        bp['totalRealCount'].fillna(0, inplace=True)
        bp['suspicious_percentage'] = bp.apply(
            lambda row: row['noEquipCount'] / row['Num_checked_patients'] * 100 if row['totalRealCount'] != 0 else 0,
            axis=1)

        anmgrouped = bp.groupby(['sub_center_id'])
        bpanm = anmgrouped.agg(
            Num_camps=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: len(list(x))),
            Tot_num_patients=pd.NamedAgg(column='Num_patients', aggfunc=sum),
            suspicious_percentage=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: list(x)),
            dates=pd.NamedAgg(column='cluster_date', aggfunc=lambda x: list(x))
        ).reset_index()

        # Step 3
        fraud_probabilities = []
        for i in range(len(bpanm)):
            fraud_prob = []
            for j in range(len(bpanm.loc[i, 'suspicious_percentage'])):
                x = bpanm.loc[i, 'suspicious_percentage'][j]
                prob = self.func_get_prob_mass_trans(self.weight_contra_fit, 0, x)
                fraud_prob.append(prob)
            fraud_probabilities.append(np.mean(fraud_prob))

        fraud_probabilities = pd.Series(fraud_probabilities)
        bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def hbcontradictionRule(self, df, dummydf):
        """
          "RULE: “No-Equipment/Entered-Data Contradiction” for haemoglobin (No Equipment and then “Present/Absent for another patient)"

          Step 1. Processes the data in the given timeframe using the rule
          Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
          Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

          Parameters
          ----------
          df : pandas DataFrame
              Dataframe to be processed
          dummydf : pandas DataFrame
              Dataframe with all the ANMs

          Returns
          -------
          list
              a list of non-diligence probabilities of ANMs
          """

        # Step 1
        grouped = df.groupby(['sub_center_id', 'camp_id', 'cluster_date'])
        bp = grouped.agg(
            Num_patients=pd.NamedAgg(column='ANC_Mother Id', aggfunc=lambda x: len(list(x))),
            Num_checked_patients=pd.NamedAgg(column='hb', aggfunc=lambda x: len(list(x.dropna()))),
            hb=pd.NamedAgg(column='hb', aggfunc=lambda x: list(x.dropna()))
        ).reset_index()

        # Step 2
        noEquipCount = []
        totalRealCount = []
        for k in range(len(bp)):
            noEquip = 0
            totReal = 0
            for i in range(len(bp.loc[k, 'hb'])):
                if (bp.loc[k, 'hb'][i] == 4):
                    noEquip += 1
                elif (bp.loc[k, 'hb'][i] > 5):
                    totReal += 1

            noEquipCount.append(noEquip)
            totalRealCount.append(totReal)

        bp = bp.assign(noEquipCount=noEquipCount)
        bp = bp.assign(totalRealCount=totalRealCount)
        bp['totalRealCount'].fillna(0, inplace=True)
        bp['suspicious_percentage'] = bp.apply(
            lambda row: row['noEquipCount'] / row['Num_checked_patients'] * 100 if row['totalRealCount'] != 0 else 0,
            axis=1)

        anmgrouped = bp.groupby(['sub_center_id'])
        bpanm = anmgrouped.agg(
            Num_camps=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: len(list(x))),
            Tot_num_patients=pd.NamedAgg(column='Num_patients', aggfunc=sum),
            suspicious_percentage=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: list(x)),
            dates=pd.NamedAgg(column='cluster_date', aggfunc=lambda x: list(x))
        ).reset_index()

        # Step 3
        fraud_probabilities = []
        for i in range(len(bpanm)):
            fraud_prob = []
            for j in range(len(bpanm.loc[i, 'suspicious_percentage'])):
                x = bpanm.loc[i, 'suspicious_percentage'][j]
                prob = self.func_get_prob_mass_trans(self.hb_contra_fit, 0, x)
                fraud_prob.append(prob)
            fraud_probabilities.append(np.mean(fraud_prob))

        fraud_probabilities = pd.Series(fraud_probabilities)
        bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def bloodsugarcontradictionRule(self, df, dummydf):
        """
          "RULE: “No-Equipment/Entered-Data Contradiction” for blood sugar (No Equipment and then “Present/Absent for another patient)"

          Step 1. Processes the data in the given timeframe using the rule
          Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
          Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

          Parameters
          ----------
          df : pandas DataFrame
              Dataframe to be processed
          dummydf : pandas DataFrame
              Dataframe with all the ANMs

          Returns
          -------
          list
              a list of non-diligence probabilities of ANMs
          """
        # Step 1
        grouped = df.groupby(['sub_center_id', 'camp_id', 'cluster_date'])
        bp = grouped.agg(
            Num_patients=pd.NamedAgg(column='ANC_Mother Id', aggfunc=lambda x: len(list(x))),
            Num_checked_patients=pd.NamedAgg(column='blood_sugar', aggfunc=lambda x: len(list(x.dropna()))),
            blood_sugar=pd.NamedAgg(column='blood_sugar', aggfunc=lambda x: list(x.dropna()))
        ).reset_index()

        # Step 2
        noEquipCount = []
        totalRealCount = []
        for k in range(len(bp)):
            noEquip = 0
            totReal = 0
            for i in range(len(bp.loc[k, 'blood_sugar'])):
                if (bp.loc[k, 'blood_sugar'][i] == 4):
                    noEquip += 1
                elif (bp.loc[k, 'blood_sugar'][i] > 7):
                    totReal += 1

            noEquipCount.append(noEquip)
            totalRealCount.append(totReal)

        bp = bp.assign(noEquipCount=noEquipCount)
        bp = bp.assign(totalRealCount=totalRealCount)
        bp['totalRealCount'].fillna(0, inplace=True)
        bp['suspicious_percentage'] = bp.apply(
            lambda row: row['noEquipCount'] / row['Num_checked_patients'] * 100 if row['totalRealCount'] != 0 else 0,
            axis=1)

        anmgrouped = bp.groupby(['sub_center_id'])
        bpanm = anmgrouped.agg(
            Num_camps=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: len(list(x))),
            Tot_num_patients=pd.NamedAgg(column='Num_patients', aggfunc=sum),
            suspicious_percentage=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: list(x)),
            dates=pd.NamedAgg(column='cluster_date', aggfunc=lambda x: list(x))
        ).reset_index()

        # Step 3
        fraud_probabilities = []
        for i in range(len(bpanm)):
            fraud_prob = []
            for j in range(len(bpanm.loc[i, 'suspicious_percentage'])):
                x = bpanm.loc[i, 'suspicious_percentage'][j]
                prob = self.func_get_prob_mass_trans(self.bsugar_contra_fit, 0, x)
                fraud_prob.append(prob)
            fraud_probabilities.append(np.mean(fraud_prob))

        fraud_probabilities = pd.Series(fraud_probabilities)
        bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def fetalhrcontradictionRule(self, df, dummydf):
        """
          "RULE: “No-Equipment/Entered-Data Contradiction” for fetal heart rate (No Equipment and then “Present/Absent for another patient)"

          Step 1. Processes the data in the given timeframe using the rule
          Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
          Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

          Parameters
          ----------
          df : pandas DataFrame
              Dataframe to be processed
          dummydf : pandas DataFrame
              Dataframe with all the ANMs

          Returns
          -------
          list
              a list of non-diligence probabilities of ANMs
          """
        # Step 1
        grouped = df.groupby(['sub_center_id', 'camp_id', 'cluster_date'])
        bp = grouped.agg(
            Num_patients=pd.NamedAgg(column='ANC_Mother Id', aggfunc=lambda x: len(list(x))),
            Num_checked_patients=pd.NamedAgg(column='fetalhr', aggfunc=lambda x: len(list(x.dropna()))),
            fetalhr=pd.NamedAgg(column='fetalhr', aggfunc=lambda x: list(x.dropna()))
        ).reset_index()

        # Step 2
        noEquipCount = []
        totalRealCount = []
        for k in range(len(bp)):
            noEquip = 0
            totReal = 0
            for i in range(len(bp.loc[k, 'fetalhr'])):
                if (bp.loc[k, 'fetalhr'][i] == 4):
                    noEquip += 1
                elif (bp.loc[k, 'fetalhr'][i] > 7):
                    totReal += 1

            noEquipCount.append(noEquip)
            totalRealCount.append(totReal)

        bp = bp.assign(noEquipCount=noEquipCount)
        bp = bp.assign(totalRealCount=totalRealCount)
        bp['totalRealCount'].fillna(0, inplace=True)
        bp['suspicious_percentage'] = bp.apply(
            lambda row: row['noEquipCount'] / row['Num_checked_patients'] * 100 if row['totalRealCount'] != 0 else 0,
            axis=1)

        anmgrouped = bp.groupby(['sub_center_id'])
        bpanm = anmgrouped.agg(
            Num_camps=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: len(list(x))),
            Tot_num_patients=pd.NamedAgg(column='Num_patients', aggfunc=sum),
            suspicious_percentage=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: list(x)),
            dates=pd.NamedAgg(column='cluster_date', aggfunc=lambda x: list(x))
        ).reset_index()

        # Step 3
        fraud_probabilities = []
        for i in range(len(bpanm)):
            fraud_prob = []
            for j in range(len(bpanm.loc[i, 'suspicious_percentage'])):
                x = bpanm.loc[i, 'suspicious_percentage'][j]
                prob = self.func_get_prob_mass_trans(self.fetalhr_contra_fit, 0, x)
                fraud_prob.append(prob)
            fraud_probabilities.append(np.mean(fraud_prob))

        fraud_probabilities = pd.Series(fraud_probabilities)
        bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def urinesugarcontradictionRule(self, df, dummydf):
        """
          "RULE: “No-Equipment/Entered-Data Contradiction” for Urine Test Sugar (No Equipment and then “Present/Absent for another patient)"

          Step 1. Processes the data in the given timeframe using the rule
          Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
          Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

          Parameters
          ----------
          df : pandas DataFrame
              Dataframe to be processed
          dummydf : pandas DataFrame
              Dataframe with all the ANMs

          Returns
          -------
          list
              a list of non-diligence probabilities of ANMs
          """
        # Step 1
        grouped = df.groupby(['sub_center_id', 'camp_id', 'cluster_date'])
        bp = grouped.agg(
            Num_patients=pd.NamedAgg(column='ANC_Mother Id', aggfunc=lambda x: len(list(x))),
            Num_checked_patients=pd.NamedAgg(column='urinesugar', aggfunc=lambda x: len(list(x.dropna()))),
            urinesugar=pd.NamedAgg(column='urinesugar', aggfunc=lambda x: list(x.dropna()))
        ).reset_index()

        # Step 2
        noEquipCount = []
        totalRealCount = []
        for k in range(len(bp)):
            noEquip = 0
            totReal = 0
            for i in range(len(bp.loc[k, 'urinesugar'])):
                if (bp.loc[k, 'urinesugar'][i] == '4'):
                    noEquip += 1
                elif (bp.loc[k, 'urinesugar'][i] == 'True') or (bp.loc[k, 'urinesugar'][i] == 'False'):
                    totReal += 1

            noEquipCount.append(noEquip)
            totalRealCount.append(totReal)

        bp = bp.assign(noEquipCount=noEquipCount)
        bp = bp.assign(totalRealCount=totalRealCount)
        bp['totalRealCount'].fillna(0, inplace=True)
        bp['suspicious_percentage'] = bp.apply(
            lambda row: row['noEquipCount'] / row['Num_checked_patients'] * 100 if row['totalRealCount'] != 0 else 0,
            axis=1)

        anmgrouped = bp.groupby(['sub_center_id'])
        bpanm = anmgrouped.agg(
            Num_camps=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: len(list(x))),
            Tot_num_patients=pd.NamedAgg(column='Num_patients', aggfunc=sum),
            suspicious_percentage=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: list(x)),
            dates=pd.NamedAgg(column='cluster_date', aggfunc=lambda x: list(x))
        ).reset_index()

        # Step 3
        fraud_probabilities = []
        for i in range(len(bpanm)):
            fraud_prob = []
            for j in range(len(bpanm.loc[i, 'suspicious_percentage'])):
                x = bpanm.loc[i, 'suspicious_percentage'][j]
                prob = self.func_get_prob_mass_trans(self.usugar_contra_fit, 0, x)
                fraud_prob.append(prob)
            fraud_probabilities.append(np.mean(fraud_prob))

        fraud_probabilities = pd.Series(fraud_probabilities)
        bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def albumincontradictionRule(self, df, dummydf):
        """
          "RULE: “No-Equipment/Entered-Data Contradiction” for Urine Test Albumin (No Equipment and then “Present/Absent for another patient)"

          Step 1. Processes the data in the given timeframe using the rule
          Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
          Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

          Parameters
          ----------
          df : pandas DataFrame
              Dataframe to be processed
          dummydf : pandas DataFrame
              Dataframe with all the ANMs

          Returns
          -------
          list
              a list of non-diligence probabilities of ANMs
          """
        # Step 1
        grouped = df.groupby(['sub_center_id', 'camp_id', 'cluster_date'])
        bp = grouped.agg(
            Num_patients=pd.NamedAgg(column='ANC_Mother Id', aggfunc=lambda x: len(list(x))),
            Num_checked_patients=pd.NamedAgg(column='albumin', aggfunc=lambda x: len(list(x.dropna()))),
            albumin=pd.NamedAgg(column='albumin', aggfunc=lambda x: list(x.dropna()))
        ).reset_index()

        # Step 2
        noEquipCount = []
        totalRealCount = []
        for k in range(len(bp)):
            noEquip = 0
            totReal = 0
            for i in range(len(bp.loc[k, 'albumin'])):
                if (bp.loc[k, 'albumin'][i] == '4'):
                    noEquip += 1
                elif (bp.loc[k, 'albumin'][i] == 'True') or (bp.loc[k, 'albumin'][i] == 'False'):
                    totReal += 1

            noEquipCount.append(noEquip)
            totalRealCount.append(totReal)

        bp = bp.assign(noEquipCount=noEquipCount)
        bp = bp.assign(totalRealCount=totalRealCount)
        bp['totalRealCount'].fillna(0, inplace=True)
        bp['suspicious_percentage'] = bp.apply(
            lambda row: row['noEquipCount'] / row['Num_checked_patients'] * 100 if row['totalRealCount'] != 0 else 0,
            axis=1)

        anmgrouped = bp.groupby(['sub_center_id'])
        bpanm = anmgrouped.agg(
            Num_camps=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: len(list(x))),
            Tot_num_patients=pd.NamedAgg(column='Num_patients', aggfunc=sum),
            suspicious_percentage=pd.NamedAgg(column='suspicious_percentage', aggfunc=lambda x: list(x)),
            dates=pd.NamedAgg(column='cluster_date', aggfunc=lambda x: list(x))
        ).reset_index()

        # Step 3
        fraud_probabilities = []
        for i in range(len(bpanm)):
            fraud_prob = []
            for j in range(len(bpanm.loc[i, 'suspicious_percentage'])):
                x = bpanm.loc[i, 'suspicious_percentage'][j]
                prob = self.func_get_prob_mass_trans(self.alb_contra_fit, 0, x)
                fraud_prob.append(prob)
            fraud_probabilities.append(np.mean(fraud_prob))

        fraud_probabilities = pd.Series(fraud_probabilities)
        bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])











