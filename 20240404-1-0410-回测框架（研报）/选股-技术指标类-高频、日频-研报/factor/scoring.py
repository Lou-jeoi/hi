import factor
import pandas as pd

def score_daily(stock):
    stock['Future_Return'] = stock['Return'].shift(-1)
    stock.dropna(how='any',inplace=True)
    Rank_ic = factor.rank_ic(stock['Future_Return'],stock.iloc[:, -2])
    Information_ratio = factor.information_ratio(Rank_ic)
    T_value = factor.t_value(Rank_ic, 252)
    #T_value = factor.t_value(Rank_ic, 252*6)
    #Win_rate = factor.win_rate(Rank_ic)
    result = [Rank_ic, Information_ratio, T_value]
    result = pd.DataFrame([result], columns=['Rank_ic', 'Information_ratio', 'T_value'])
    return result

