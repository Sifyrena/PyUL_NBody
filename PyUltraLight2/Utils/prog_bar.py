from PyUltraLight2.Utils.printU import *
import time
from datetime import datetime

####################### AUX. FUNCTION TO GENERATE PROGRESS BAR
def prog_bar(iteration_number=100, progress=1, tinterval=0,
             status='', adtl='', OutNumber=0):
    
    width = 30  # wider = smoother perception
    
    # --- Progress ---
    prog = min(max(float(progress) / float(iteration_number), 0.0), 1.0)
    
    # Sub-block resolution (8 levels)
    full_blocks = int(prog * width)
    remainder = (prog * width) - full_blocks
    partial_block = int(remainder * 8)

    blocks = "█" * full_blocks
    if full_blocks < width:
        partial_chars = "▏▎▍▌▋▊▉"
        part = partial_chars[partial_block - 1] if partial_block > 0 else ""
    else:
        part = ""
    
    empty = " " * (width - full_blocks - (1 if part else 0))
    
    bar = f"{blocks}{part}{empty}"
    
    # --- ETA ---
    if tinterval > 0 and progress < iteration_number:
        eta_seconds = (iteration_number - progress) * tinterval
        eta_stamp = time.time() + eta_seconds
        ETA = datetime.fromtimestamp(eta_stamp).strftime("%H:%M:%S")
    else:
        ETA = "--:--:--"
    
    # --- Labels ---
    if OutNumber != 0:
        prefix = f"({OutNumber:03d})"
    else:
        prefix = f"{prog * 100:6.2f}%"
    
    # --- Timing ---
    timing = f"{tinterval:6.2f}s"
    
    # --- Status formatting ---
    status = status.strip()
    if status:
        status = f"| {status}"
    
    if adtl:
        adtl = f"| {adtl}"
    
    # --- Final string ---
    text = (
        f"\r[{bar}] "
        f"{prefix} "
        f"ETA {ETA} "
        f"Δt {timing} "
        f"{status} {adtl}"
    )
    
    print(text, end="", flush=True)
    
    
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

