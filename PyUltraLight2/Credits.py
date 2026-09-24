from .Version import Version, D_version, S_version
####################### Credits Information
def Credits(IsoP = False,UseDispSponge = False,embeds = [], SI = False):
    print(f"==============================================================================")
    print(f"{Version} v{S_version}: (c) 2020 - 2024 Yourong F. Wang and collaborators. \nAuckland Cosmology\n")
    print(D_version)
    print("Original PyUltraLight Team:\nEdwards, F., Kendall, E., Hotchkiss, S. & Easther, R.\n\
arxiv.org/abs/1807.04037")
    
    if IsoP or UseDispSponge or (embeds != []):
        print("\n**External Module In Use**")
    
    if IsoP:
        print("\nIsolated ULDM Potential Implementation \nAdapted from J. L. Zagorac et al.")
        
    if UseDispSponge:
        print("\nDispersive Sponge Condition \nAdapted from J. L. Zagorac et al.")
        
    if embeds != []:
        print("\nEmbedded Soliton Profiles \nAdapted from N. Guo et al.")
        
    if SI:
        print("\nULDM Self Interaction Enabled as a Quartic Extra Potential:\nBased on Stallovits, M. and Rindler-Daller T. \n\
arxiv.org/abs/2406.07419"
        )
    print(f"==============================================================================")
