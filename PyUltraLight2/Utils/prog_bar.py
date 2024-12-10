from PyUltraLight2.Utils.printU import *
import time
from datetime import datetime

####################### AUX. FUNCTION TO GENERATE PROGRESS BAR
def prog_bar(iteration_number = 100, progress = 1, tinterval = 0 ,status = '',adtl = '', OutNumber = 0):
    size = 10
    
    if tinterval != 0:
        ETAStamp = time.time() + (iteration_number - progress)*tinterval

        ETA = datetime.fromtimestamp(ETAStamp).strftime("%d/%m/%Y, %H:%M:%S")
    
    else:
        ETA = '-'
        
    PROG = float(progress) / float(iteration_number)
    
    if PROG >= 1.:
        PROG, status = 1, ""
    
    block = int(round((size) * PROG))
    
    status = f'{status.ljust(SSLength)}'
  
    if block == 0:
        CM = '◎'
        PM = ''
        PL = 0
    else:
        CM = '○'
        PM = '◎' 
        PL = 1
    
    if block == size:
        PM = CM
        
    CL = 2*block - 2*PL + 1
    
    LL = size - block
    RL = size - block

    BarText = "●" * LL + PM * PL + CM * CL + PM * PL + "●" * RL
    
    shift = int(-1 * progress % (2*size+1))

    BarText = BarText[shift:-1] + BarText[0:shift]
    
    if OutNumber != 0:
        SaveText = f"({OutNumber:03d})"
    else:
        SaveText = f"{PROG * 100:.0f}%"
    
    text = "\r[{}] {} {}{}{} ({}{:.2f}s) {}".format(BarText
        ,
        SaveText,
        status, 'ETA: ',ETA,'Prev.: ',tinterval,adtl)
   
    print(f'{text}', end="",flush='true')
    
    
def prog_bar_NG(iteration_number = 100, progress = 1, tinterval = 0 ,status = '',adtl = ''):
        
    if tinterval != 0:
        ETAStamp = time.time() + (iteration_number - progress)*tinterval

        ETA = datetime.fromtimestamp(ETAStamp).strftime("%d/%m/%Y, %H:%M:%S")
    
    else:
        ETA = '-'
        
    PROG = float(progress) / float(iteration_number)
    
    if PROG >= 1.:
        PROG, status = 1, ""
    
    text = f"{round(PROG * 100, 0)},{status}, Exp. Time: {ETA},'Prev.: ',{tinterval},{adtl}"
   
    printU(text,"")

