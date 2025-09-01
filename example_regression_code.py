# -*- coding: utf-8 -*-
"""
Terner Labs - Economist Data Assignment (Section B)

Example code snippet: 2023 Dissertation

@author: Sarah Neubauer


"""

# Dissertation Example: Evacuation Behavior & Social Connectedness

# 0. Packages
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

# 1. Load regression data
reg_dat = pd.read_csv("reg_dat.csv")

# 2. Subsets
reg_intra_more = reg_dat[reg_dat["type"] == "evacuation_intra_more"] # More long distance movement
reg_inter_less = reg_dat[reg_dat["type"] == "evacuation_inter_less"] #Less local movement
reg_more = reg_dat[reg_dat["type"].isin(["evacuation_inter_more", "evacuation_intra_more"])] #Overall more movement
reg_less = reg_dat[reg_dat["type"].isin(["evacuation_inter_less", "evacuation_intra_less"])] #Overall less movement

# 3. Basic regressions
r1 = smf.ols("evacuation_per_thous ~ log_sci_csd", data=reg_dat).fit()
r2 = smf.ols("evacuation_per_thous ~ log_sci_csd", data=reg_intra_more).fit()
r3 = smf.ols("evacuation_per_thous ~ log_sci_csd", data=reg_inter_less).fit()
r4 = smf.ols("evacuation_per_thous ~ log_sci_csd", data=reg_more).fit()
r5 = smf.ols("evacuation_per_thous ~ log_sci_csd", data=reg_less).fit()

#Example output
print(r1.summary())

# 4. Add controls (social capital)
controls1 = "bonding_user + bridging_user + linking_user + bonding_fr + bridging_fr + linking_fr"
r1_2 = smf.ols(f"evacuation_per_thous ~ log_sci_csd + {controls1}", data=reg_dat).fit()
r2_2 = smf.ols("evacuation_per_thous ~ log_sci_csd + bonding_user + bridging_user + linking_user", data=reg_intra_more).fit()
r3_2 = smf.ols(f"evacuation_per_thous ~ log_sci_csd + {controls1}", data=reg_inter_less).fit()
r4_2 = smf.ols(f"evacuation_per_thous ~ log_sci_csd + {controls1}", data=reg_more).fit()
r5_2 = smf.ols(f"evacuation_per_thous ~ log_sci_csd + {controls1}", data=reg_less).fit()

#Example output
print(r2_2.summary())





