
import os
import dataGet_Func
import pandas as pd
import factor
import scoring

os.chdir("D:/实习/CCBFinTech/No.26")

'''
import tushare as ts
df_hf = ts.pro_bar(ts_code='000985.SH',freq='1min', start_date='2013-01-01 09:00:00', end_date='2013-12-31 17:00:00')
df_hf.to_pickle('./data/000985_level2_1min_2013.pickle')
'''

file = open('./data/RESSET_STKSZ2022_002336_002536_1_1.csv')
df_b = pd.read_csv(file,low_memory=False)
# filec = open('./data/RESSET_DRESSTK_2021__1.csv')
# base = pd.read_csv(filec,low_memory=False)

#stock = stock[(stock['Stdtime'] >= '09:31:00') & (stock['Stdtime'] <= '14:56:00') & (~stock['Stdtime'].between('11:31:00', '13:01:00'))]
#stock = stock[(~stock['Stdtime'].between('9:26:00', '9:30:00'))&(~stock['Stdtime'].between('11:31:00', '13:00:00'))&(~stock['Stdtime'].between('14:56:00', '15:30:00'))]
#stocks = stock[(~stock['Stdtime'].between('9:26', '9:31')) & (~stock['Stdtime'].between('11:31', '13:00')) & (~stock['Stdtime'].between('14:56', '15:30'))]


#rvs_hf = factor.reverse_hf(df_b)
rvs_daily = factor.reverse_daily(df_b)
#rvs_imp_pos = factor.reverse_imp_pos(df_b)
#rvs_imp_neg = factor.reverse_imp_neg(df_b)
#ret_std_hf = factor.return_std_hf(df_b)
#ret_std_daily = factor.return_std_daily(df_b)
#ret_std_imp_daily = factor.return_std_imp_daily(df_b)
#t = factor.tail_volume_factor(df_b,base)
a = scoring.score_daily(rvs_daily)

# file = open('D:/实习\CCBFinTech/20240404-1-0410-回测框架（研报）111/选股-技术指标类-高频、日频-研报（范例）/result/RVar_signal_result_all_config.pickle',encoding='utf-8')
# df = dataGet_Func.read_file('D:/实习\CCBFinTech/20240404-1-0410-回测框架（研报）111/选股-技术指标类-高频、日频-研报（范例）/result/config0/portfolio0',file_name='portfolio0_attributes_security_df.pickle')