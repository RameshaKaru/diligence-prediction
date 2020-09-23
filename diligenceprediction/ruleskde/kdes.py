import rpy2.robjects as robjects


class KDEs:
    """
    This class serves as the intermediary between R objects and Python.

    At initialization, the KDEs are obtained using R scripts.

    """

    def __init__(self):
        robjects.r('''
            source('diligenceprediction/rfunc/helpers.r')
        ''')

        self.func_get_prob_mass_trans = robjects.globalenv['func_get_prob_mass_trans']

        robjects.r('''
            source('diligenceprediction/rfunc/kdedata.r')
        ''')

        self.death_fit = robjects.globalenv['death_fit']
        self.ebf_fit = robjects.globalenv['ebf_fit']
        self.hiv_fit = robjects.globalenv['hiv_fit']
        self.bloodg_fit = robjects.globalenv['bloodg_fit']
        self.vdrl_fit = robjects.globalenv['vdrl_fit']
        self.fundalh_fit = robjects.globalenv['fundalh_fit']

        self.bpsd_fit = robjects.globalenv['bpsd_fit']
        self.urine_fit = robjects.globalenv['urine_fit']
        self.hb50_fit = robjects.globalenv['hb50_fit']
        self.hb70_fit = robjects.globalenv['hb70_fit']
        self.hb7_fit = robjects.globalenv['hb7_fit']
        self.bphyper_fit = robjects.globalenv['bphyper_fit']
        self.bp_contra_fit = robjects.globalenv['bp_contra_fit']
        self.weight_contra_fit = robjects.globalenv['weight_contra_fit']
        self.hb_contra_fit = robjects.globalenv['hb_contra_fit']
        self.bsugar_contra_fit = robjects.globalenv['bsugar_contra_fit']
        self.fetalhr_contra_fit = robjects.globalenv['fetalhr_contra_fit']
        self.usugar_contra_fit = robjects.globalenv['usugar_contra_fit']
        self.alb_contra_fit = robjects.globalenv['alb_contra_fit']