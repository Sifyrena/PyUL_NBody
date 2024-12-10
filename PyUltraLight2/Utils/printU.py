import builtins

SSLength = 7

from PyUltraLight2.Version import *
from PyUltraLight2.Utils.IO import GenFromTime

def printU(Message,SubSys = 'Sys',ToScreen = True, ToFile = False, FilePath = ''):

    if ToScreen:
        builtins.print(f"{Version}{SubSys.rjust(SSLength)}: {Message}")
        
    if ToFile and FilePath != "":
        with open(FilePath, "a+") as o:
            o.write(f"{GenFromTime()} {Version}.{SubSys.rjust(SSLength)}: {Message}\n")
