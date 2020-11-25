# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 22:04:20 2020

@author: Peilin Yang
"""

import pandas as pd
import numpy as np
import datetime

def cal_smb_hml(df):
    # according to circ_mv: Market Value
    df['SB'] = df['circ_mv'].map(lambda x: 'B' if x >= df['circ_mv'].median() else 'S')
    # Book Market Value
    df['BM'] = 1 / df['pb']
    
    # H M L
    border_down, border_up = df['BM'].quantile([0.3, 0.7])
    df['HML'] = df['BM'].map(lambda x: 'H' if x >= border_up else 'M')
    df['HML'] = df.apply(lambda x: 'L' if x['BM'] <= border_down else x['HML'], axis=1)
    
    df_SL = df.query('(SB=="S") & (HML=="L")')
    df_SM = df.query('(SB=="S") & (HML=="M")')
    df_SH = df.query('(SB=="S") & (HML=="H")')
    df_BL = df.query('(SB=="B") & (HML=="L")')
    df_BM = df.query('(SB=="B") & (HML=="M")')
    df_BH = df.query('(SB=="B") & (HML=="H")')
    # Market Weighted Returns：ΣReturn × Cir Market Value/Σ Cir Market Value
    R_SL = (df_SL['pct_chg'] * df_SL['circ_mv'] / 100).sum() / df_SL['circ_mv'].sum()
    R_SM = (df_SM['pct_chg'] * df_SM['circ_mv'] / 100).sum() / df_SM['circ_mv'].sum()
    R_SH = (df_SH['pct_chg'] * df_SH['circ_mv'] / 100).sum() / df_SH['circ_mv'].sum()
    R_BL = (df_BL['pct_chg'] * df_BL['circ_mv'] / 100).sum() / df_BL['circ_mv'].sum()
    R_BM = (df_BM['pct_chg'] * df_BM['circ_mv'] / 100).sum() / df_BM['circ_mv'].sum()
    R_BH = (df_BH['pct_chg'] * df_BH['circ_mv'] / 100).sum() / df_BH['circ_mv'].sum()

    smb = (R_SL + R_SM + R_SH - R_BL - R_BM - R_BH) / 3
    hml = (R_SH + R_BH - R_SL - R_BL) / 2
    return smb, hml


# Fama-French Factors Coefficient
df.maotai=df.maotai-rf
df.gzA=df.gzA-rf
import statsmodels.api as sm
model=sm.OLS(df['maotai'],sm.add_constant(df[['gzA', 'SMB', 'HML']])).fit()