from rpy2.rinterface import RRuntimeError
from rpy2.robjects.packages import importr
utils = importr('utils')

def importr_tryhard(packname):
    try:
        rpack = importr(packname)
    except RRuntimeError:
        utils.install_packages(packname)
        rpack = importr(packname)
    return rpack

importr_tryhard("kde1d")