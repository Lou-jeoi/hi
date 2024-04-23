import numpy as np
import pandas as pd
from scipy.stats import rankdata
import statsmodels.api as sm

#因子部分
def reverse_hf(df):
    """
    计算收益反转因子
    :param df: DataFrame, 包含股票分钟线数据，至少包含列：['Code_Mkt','QTime','Stdtime', 'Ret_Min']
    :return: DataFrame, 包含股票的收益反转因子，列包括：['code', 'QTime','Stdtime', 'reverse_hf']
    """
    # 选择需要的列并重命名
    # stock = df[['Code_Mkt', 'QTime', 'Stdtime', 'Ret_Min']].rename(columns={'Code_Mkt': 'code', 'QTime': 'QTime', 'Stdtime': 'Stdtime', 'Ret_Min':'Ret_Min'})
    stock = df[['代码与市场标识_Code_Mkt', '行情日期_Qdate', '标准时间_StdTime', '期间收益率_Ret_Min', '期间累计成交量(股)_TVolume_accu1']].copy()
    stock.rename(columns={'代码与市场标识_Code_Mkt': 'Code', '行情日期_Qdate': 'Qdate', '标准时间_StdTime': 'StdTime',
                          '期间收益率_Ret_Min': 'Ret_Min', '期间累计成交量(股)_TVolume_accu1': 'volume'}, inplace=True)

    stock = stock[(~stock['StdTime'].between('9:26', '9:31')) & (~stock['StdTime'].between('11:31', '13:01')) & (
        ~stock['StdTime'].between('14:56', '15:31'))]

    # 将Qdate列转换为日期时间类型
    stock['Qdate'] = pd.to_datetime(stock['Qdate'], format='%Y/%m/%d')
    # 将StdTime列转换为日期时间类型
    stock['StdTime'] = pd.to_datetime(stock['StdTime'], format="%H:%M:%S")


    stock['total_minutes'] = stock['StdTime'].dt.hour * 60 + stock['StdTime'].dt.minute - 1

    # 计算半小时频次收益率、累计收益率
    stock['Reverse_hf'] = - stock.groupby(['Code', 'Qdate',stock['total_minutes'] // 30])['Ret_Min'].transform('sum')

    stock['Return'] = stock.groupby(['Code', 'Qdate', stock['total_minutes'] // 30])[
        'Ret_Min'].transform(lambda x: (1 + x).prod() - 1)

    # 保留每只股票每半小时的最后一支观测
    stock = stock.groupby(['Code', 'Qdate',stock['total_minutes'] // 30]).tail(1)

    stock['StdTime'] = stock['StdTime'].dt.strftime('%H:%M')
    rows_to_drop = stock[(stock['StdTime'] == '09:30') | (stock['StdTime'] == '14:55')].index
    stock = stock.drop(rows_to_drop)

    return stock[['Code', 'Qdate','Return', 'Reverse_hf']]


def reverse_daily(df):
    """
        计算每日收益反转因子
        :param df: DataFrame, 包含股票分钟线数据，至少包含列：['Code_Mkt', 'QTime', 'Stdtime', 'Ret_Min']
        :return result: DataFrame, 包含股票的收益反转因子，列包括：['code', 'QTime', 'Stdtime', 'reverse_daily']
        """

    # 选择需要的列并重命名
    #stock = df[['Code_Mkt', 'QTime', 'Stdtime', 'Ret_Min']].rename(columns={'Code_Mkt': 'code', 'QTime': 'QTime', 'Stdtime': 'Stdtime', 'Ret_Min':'Ret_Min'})
    stock = df[['代码与市场标识_Code_Mkt', '行情日期_Qdate', '标准时间_StdTime', '期间收益率_Ret_Min', '期间累计成交量(股)_TVolume_accu1']].copy()
    stock.rename(columns={'代码与市场标识_Code_Mkt': 'Code', '行情日期_Qdate': 'Qdate', '标准时间_StdTime': 'StdTime',
                          '期间收益率_Ret_Min': 'Ret_Min', '期间累计成交量(股)_TVolume_accu1': 'volume'}, inplace=True)

    stock = stock[(~stock['StdTime'].between('9:26', '9:31')) & (~stock['StdTime'].between('11:31', '13:01')) & (~stock['StdTime'].between('14:56', '15:31'))]

    # 将Qdate列转换为日期时间类型
    stock['Qdate'] = pd.to_datetime(stock['Qdate'], format='%Y/%m/%d')
    # 将StdTime列转换为日期时间类型
    stock['StdTime'] = pd.to_datetime(stock['StdTime'], format="%H:%M:%S")

    stock['Reverse_daily'] = - stock.groupby(['Code', 'Qdate'])['Ret_Min'].transform('sum')

    # 计算日度收益率
    stock['Return'] = stock.groupby(['Code', 'Qdate'])['Ret_Min'].transform(lambda x: (1 + x).prod() - 1)

    # 保留每只股票每天的最后一支观测
    stock = stock.groupby(['Code', stock['Qdate'].dt.date]).tail(1)

    return stock[['Code', 'Qdate','Return', 'Reverse_daily']]

def reverse_imp_pos(df):
    """
    计算反转因子（日度）

    Parameters:
    df (DataFrame): 包含股票分钟线数据的数据框，至少包含列：['Code_Mkt', 'Qdate', 'Ret_Min', 'TVolume_accu1']

    Returns:
    DataFrame: 包含股票的反转因子（日度），列包括：['Code', 'Qdate', 'Reverse_Imp_pos']
    """
    #df['Code'] = df['Code'].astype(str)

    # 选择需要的列并重命名
    stock = df[['代码与市场标识_Code_Mkt', '行情日期_Qdate','标准时间_StdTime', '期间收益率_Ret_Min', '期间累计成交量(股)_TVolume_accu1']].copy()
    stock.rename(columns={'代码与市场标识_Code_Mkt':'Code','行情日期_Qdate':'Qdate','标准时间_StdTime':'StdTime','期间收益率_Ret_Min': 'Ret_Min', '期间累计成交量(股)_TVolume_accu1': 'volume'}, inplace=True)

    stock = stock[(~stock['StdTime'].between('9:26', '9:31')) & (~stock['StdTime'].between('11:31', '13:00')) & (
        ~stock['StdTime'].between('14:56', '15:31'))]

    # 将Qdate列转换为日期时间类型
    stock['Qdate'] = pd.to_datetime(stock['Qdate'], format='%Y/%m/%d')


    # 计算每只股票每日的交易量均值
    stock['vol_mean'] = stock.groupby(['Code', 'Qdate'])['volume'].transform('mean')
    # 计算每只股票每日的交易量标准差
    stock['vol_std'] = stock.groupby(['Code', 'Qdate'])['volume'].transform('std')

    # 计算反转因子
    stock['vol_up'] = stock['vol_mean'] + stock['vol_std']
    stock['positive_return'] = (stock['Ret_Min'] > 0).astype(int)
    stock['above_vol_up'] = (stock['volume'] > stock['vol_up']).astype(int)
    stock['weighted_return'] = stock['Ret_Min'] * stock['positive_return'] * stock['above_vol_up']
    stock['denominator'] = stock['positive_return'] * stock['above_vol_up']

    # 按日期分组计算反转因子
    stock['Reverse_Imp_pos'] = -stock.groupby(stock['Qdate'].dt.date)['weighted_return'].transform('sum') / stock.groupby(stock['Qdate'].dt.date)['denominator'].transform('sum')

    # 计算日度收益率
    stock['Return'] = stock.groupby(['Code', 'Qdate'])['Ret_Min'].transform(lambda x: (1 + x).prod() - 1)

    # 保留每只股票每天的最后一支观测
    stock = stock.groupby(['Code', stock['Qdate'].dt.date]).tail(1)

    return stock[['Code', 'Qdate','Return', 'Reverse_Imp_pos']]


def reverse_imp_neg(df):
    """
    计算反转因子（日度）

    Parameters:
    df (DataFrame): 包含股票分钟线数据的数据框，至少包含列：['Code_Mkt', 'Qdate', 'Ret_Min', 'TVolume_accu1']

    Returns:
    DataFrame: 包含股票的反转因子（日度），列包括：['Code', 'Qdate', 'Reverse_Imp_pos']
    """
    #df['Code'] = df['Code'].astype(str)

    # 选择需要的列并重命名
    stock = df[['代码与市场标识_Code_Mkt', '行情日期_Qdate','标准时间_StdTime', '期间收益率_Ret_Min', '期间累计成交量(股)_TVolume_accu1']].copy()
    stock.rename(columns={'代码与市场标识_Code_Mkt':'Code','行情日期_Qdate':'Qdate','标准时间_StdTime':'StdTime','期间收益率_Ret_Min': 'Ret_Min', '期间累计成交量(股)_TVolume_accu1': 'volume'}, inplace=True)

    stock = stock[(~stock['StdTime'].between('9:26', '9:31')) & (~stock['StdTime'].between('11:31', '13:00')) & (
        ~stock['StdTime'].between('14:56', '15:31'))]

    # 将Qdate列转换为日期时间类型
    stock['Qdate'] = pd.to_datetime(stock['Qdate'], format='%Y/%m/%d')

    # 计算每只股票每日的交易量均值
    stock['vol_mean'] = stock.groupby(['Code', 'Qdate'])['volume'].transform('mean')
    # 计算每只股票每日的交易量标准差
    stock['vol_std'] = stock.groupby(['Code', 'Qdate'])['volume'].transform('std')

    # 计算反转因子
    stock['vol_up'] = stock['vol_mean'] + stock['vol_std']
    stock['negative_return'] = (stock['Ret_Min'] < 0).astype(int)
    stock['above_vol_up'] = (stock['volume'] > stock['vol_up']).astype(int)
    stock['weighted_return'] = stock['Ret_Min'] * stock['negative_return'] * stock['above_vol_up']
    stock['denominator'] = stock['negative_return'] * stock['above_vol_up']

    # 按日期分组计算反转因子
    stock['Reverse_Imp_neg'] = -stock.groupby(stock['Qdate'].dt.date)['weighted_return'].transform('sum') / stock.groupby(stock['Qdate'].dt.date)['denominator'].transform('sum')

    # 计算日度收益率
    stock['Return'] = stock.groupby(['Code', 'Qdate'])['Ret_Min'].transform(lambda x: (1 + x).prod() - 1)

    # 保留每只股票每天的最后一支观测
    stock = stock.groupby(['Code', stock['Qdate'].dt.date]).tail(1)

    return stock[['Code', 'Qdate','Return', 'Reverse_Imp_neg']]


def return_std_hf(df):
    """
    计算基于分钟线的收益反转因子（高频）

    Parameters:
    df (DataFrame): 包含股票分钟线数据的数据框，至少包含列：['代码与市场标识_Code_Mkt', '行情时间_QTime','标准时间_StdTime', '期间收益率_Ret_Min']

    Returns:
    DataFrame: 包含股票的收益反转因子（高频），列包括：['Code', 'QTime', 'Return_Std_high_freq']
    """
    # 选择需要的列并重命名
    stock = df[['代码与市场标识_Code_Mkt', '行情时间_QTime','标准时间_StdTime','期间收益率_Ret_Min']].copy()
    stock.rename(columns={'代码与市场标识_Code_Mkt':'Code','行情时间_QTime':'QTime','标准时间_StdTime':'StdTime','期间收益率_Ret_Min': 'Ret_Min'}, inplace=True)

    stock = stock[(~stock['StdTime'].between('9:26', '9:31')) & (~stock['StdTime'].between('11:31', '13:00')) & (
        ~stock['StdTime'].between('15:01', '15:31'))]

    # 将QTime、StdTime列转换为日期时间类型
    stock['QTime'] = pd.to_datetime(stock['QTime'], format='%Y-%m-%d %H:%M:%S')
    stock['StdTime'] = pd.to_datetime(stock['StdTime'], format="%H:%M:%S")

    stock['StdTime_move1'] = pd.to_datetime(stock['StdTime'], format="%H:%M") - pd.Timedelta(minutes=1)  #使整点归入上一区间

    # 计算每半小时时段的收益标准差
    stock['Return_Std_high_freq'] = stock.groupby([stock['QTime'].dt.date, stock['StdTime'].dt.hour, stock['StdTime_move1'].dt.minute // 30])[
        'Ret_Min'].transform(
        lambda x: -np.sqrt(np.mean((x - x.mean()) ** 2))
    )

    stock['total_minutes'] = stock['StdTime'].dt.hour * 60 + stock['StdTime'].dt.minute - 1

    # 计算半小时频次收益率
    stock['Return'] = stock.groupby(['Code',stock['QTime'].dt.date,stock['total_minutes']// 30])['Ret_Min'].transform(lambda x: (1 + x).prod() - 1)

    # 保留每只股票每天每半小时的最后一支观测
    stock = stock.groupby(
        ['Code', stock['QTime'].dt.date, stock['total_minutes']// 30]).tail(1)

    stock['StdTime'] = stock['StdTime'].dt.strftime('%H:%M')
    stock = stock[(stock['StdTime'] != '09:30') & (stock['StdTime'] != '15:00')]

    return stock[['Code', 'QTime', 'StdTime','Return', 'Return_Std_high_freq']].drop_duplicates()

def return_std_daily(df):
    """
    计算基于日度的收益波动率因子

    Parameters:
    df (DataFrame): 包含股票分钟线数据的数据框，至少包含列：['代码与市场标识_Code_Mkt', '行情日期_Qdate','标准时间_StdTime', '期间收益率_Ret_Min']

    Returns:
    DataFrame: 包含股票的收益波动率因子（日度），列包括：['Code', 'Qdate', 'Return_Std_daily']
    """
    # 选择需要的列并重命名
    stock = df[['代码与市场标识_Code_Mkt', '行情日期_Qdate', '标准时间_StdTime', '期间收益率_Ret_Min']].copy()
    stock.rename(columns={'代码与市场标识_Code_Mkt': 'Code', '行情日期_Qdate': 'Qdate', '标准时间_StdTime': 'StdTime',
                          '期间收益率_Ret_Min': 'Ret_Min'}, inplace=True)

    stock = stock[(~stock['StdTime'].between('9:26', '9:31')) & (~stock['StdTime'].between('11:31', '13:00')) & (
        ~stock['StdTime'].between('14:56', '15:31'))]

    # 将QTime、StdTime列转换为日期时间类型
    stock['Qdate'] = pd.to_datetime(stock['Qdate'], format='%Y-%m-%d')
    stock['StdTime'] = pd.to_datetime(stock['StdTime'], format="%H:%M:%S")

    stock['Qdate_str'] = stock['Qdate'].dt.strftime('%Y-%m-%d')

    # 计算日度收益率
    stock['Return'] = stock.groupby(['Code', 'Qdate'])['Ret_Min'].transform(lambda x: (1 + x).prod() - 1)

    # 计算每日收益标准差
    stock['Return_Std_daily'] = stock.groupby('Qdate_str')['Return'].transform(
        lambda x: -np.sqrt(np.mean((x - x.mean()) ** 2)))

    # 保留每只股票每天的最后一支观测
    stock = stock.groupby(['Code', stock['Qdate'].dt.date]).tail(1)

    return stock[['Code','Qdate','Return','Return_Std_daily']]


def return_std_imp_daily(df):
    """
    计算基于日度的改进收益波动率因子

    Parameters:
    df (DataFrame): 包含股票分钟线数据的数据框，至少包含列：['代码与市场标识_Code_Mkt', '行情日期_Qdate','标准时间_StdTime', '期间收益率_Ret_Min', '期间累计成交量(股)_TVolume_accu1']

    Returns:
    DataFrame: 包含股票的改进收益波动率因子（日度），列包括：['Code', 'Qdate', 'Return_Std_impi_daily']
    """
    # 选择需要的列并重命名
    stock = df[['代码与市场标识_Code_Mkt', '行情日期_Qdate', '标准时间_StdTime', '期间收益率_Ret_Min', '期间累计成交量(股)_TVolume_accu1']].copy()
    stock.rename(columns={'代码与市场标识_Code_Mkt': 'Code', '行情日期_Qdate': 'Qdate', '标准时间_StdTime': 'StdTime',
                          '期间收益率_Ret_Min': 'Ret_Min', '期间累计成交量(股)_TVolume_accu1': 'Volume'}, inplace=True)

    stock = stock[(~stock['StdTime'].between('9:26', '9:31')) & (~stock['StdTime'].between('11:31', '13:00')) & (
        ~stock['StdTime'].between('14:56', '15:31'))]

    # 将QTime、StdTime列转换为日期时间类型
    stock['Qdate'] = pd.to_datetime(stock['Qdate'], format='%Y-%m-%d')
    stock['StdTime'] = pd.to_datetime(stock['StdTime'], format="%H:%M:%S")

    # 将Qdate转换为日期字符串
    stock['Qdate_str'] = stock['Qdate'].dt.strftime('%Y-%m-%d')

    stock['Volume_Mean'] = stock.groupby(['Code','Qdate_str'])['Volume'].transform('mean')
    stock['Volume_Std'] = stock.groupby(['Code','Qdate_str'])['Volume'].transform('std')

    stock['Vol_up'] = stock['Volume_Mean'] + stock['Volume_Std']

    # 筛选放量时段(标记成交量是否超过阈值)
    stock['Volume_Flag'] = (stock['Volume'] > stock['Vol_up']).astype(int)

    # 计算每天成交量超过阈值的次数
    stock['Volume_Above_Count'] = stock.groupby(['Code', 'Qdate_str'])['Volume_Flag'].transform('sum')

    #计算每天成交量超过阈值时段内的累计成交量
    stock['y'] = stock['Volume'] * stock['Volume_Above_Count']
    stock['Volume_Above_Accu'] = stock.groupby(['Code', 'Qdate_str'])['y'].transform('sum')

    stock['R_Mean'] = stock.groupby(['Code','Qdate_str'])['Ret_Min'].transform('mean')
    #stock['R_Mean'] = stock['Volume_Above_Accu']/stock['Volume_Above_Count']            #研报文字与公式表述不一致

    stock['Daily_Return_Mean'] = stock.groupby(['Code','Qdate_str'])['Ret_Min'].transform('mean')
    stock['Daily_Return_Std'] = stock.groupby(['Code','Qdate_str'])['Ret_Min'].transform('std')

    stock['x'] = (stock['Ret_Min'] - stock['R_Mean']) ** 2/ stock['Volume_Above_Count']
    stock['Return_Std_imp_daily'] = - np.sqrt(stock.groupby(['Code','Qdate_str'])['x'].transform('sum'))

    # 计算日度收益率
    stock['Return'] = stock.groupby(['Code', 'Qdate'])['Ret_Min'].transform(lambda x: (1 + x).prod() - 1)

    # 保留每只股票每天的最后一支观测
    stock = stock.groupby(['Code', stock['Qdate'].dt.date]).tail(1)

    return stock[['Code','Qdate','Return','Return_Std_imp_daily']]


def tail_volume_factor(df1,df2):
    """
    计算尾盘成交额因子

    Parameters:
    df1 (DataFrame): 包含股票分钟线数据的数据框，至少包含列：['代码与市场标识_Code_Mkt', '行情日期_Qdate', '标准时间_StdTime', '成交额','期间收益率_Ret_Min']
    df2 (DataFrame): 股票综合数据，至少包含列：['股票代码_Stkcd','日期_Date','收盘价_Clpr','流通股_Trdshr']（或流通市值）

    Returns:
    DataFrame: 包含股票的尾盘成交额因子，列包括：['Code', 'Qdate', 'Tail_Volume_Factor']
    """
    # 选择需要的列并重命名
    stock = df1[['代码与市场标识_Code_Mkt', '行情日期_Qdate', '标准时间_StdTime', '期间累计成交额(元)_TSum_accu1','期间收益率_Ret_Min']].copy()
    base = df2[['股票代码_Stkcd','日期_Date','收盘价_Clpr','流通股_Trdshr']].copy()

    stock.rename(columns={'代码与市场标识_Code_Mkt': 'Code', '行情日期_Qdate': 'Qdate', '标准时间_StdTime': 'StdTime',
                          '期间累计成交额(元)_TSum_accu1': 'TrdSum','期间收益率_Ret_Min':'Ret_Min'}, inplace=True)

    base.rename(columns={'股票代码_Stkcd':'Code','日期_Date':'Qdate','收盘价_Clpr':'Clpr','流通股_Trdshr':'Trdshr'}, inplace=True)

    stock = stock[(~stock['StdTime'].between('9:26', '9:31')) & (~stock['StdTime'].between('11:31', '13:00')) & (
        ~stock['StdTime'].between('14:56', '15:31'))]

    # 将QTime、StdTime列转换为日期时间类型
    stock['Qdate'] = pd.to_datetime(stock['Qdate'], format='%Y-%m-%d')
    stock['StdTime'] = pd.to_datetime(stock['StdTime'], format="%H:%M:%S")
    base['Qdate'] = pd.to_datetime(base['Qdate'], format='%Y/%m/%d')


    # 将Qdate转换为日期字符串
    stock['Qdate_str'] = stock['Qdate'].dt.strftime('%Y-%m-%d')

    # 计算T-1日的流通市值
    base['TMV'] = base['Clpr'] * base['Trdshr']
    base['TMV_back'] = base['TMV'].shift(1)

    #14：30-14：55（倒数26个观测）成交额组成当日尾盘成交额
    stock['x'] = stock.groupby(['Code','Qdate'])['TrdSum'].tail(26)
    stock['tail_TrdSum']= stock.groupby(['Code','Qdate'])['x'].transform('sum')

    stock = pd.merge(stock,base,on=['Code','Qdate'])

    # 计算尾盘成交额因子
    stock['ttv_Ratio'] = stock['tail_TrdSum'] / stock['TMV_back']

    # 计算日度收益率
    stock['Return'] = stock.groupby(['Code', 'Qdate'])['Ret_Min'].transform(lambda x: (1 + x).prod() - 1)

    # 保留每只股票每天的最后一支观测
    stock = stock.groupby(['Code', stock['Qdate'].dt.date]).tail(1)
    stock.dropna(how='any',inplace=True)

    return stock[['Code','Qdate','Return','ttv_Ratio']]



def factor_neutralization(stock, market_values, industry_dummies):
    """
    对因子进行市值和行业中性化处理
    Args:
        factor_values (DataFrame): 因子值的 DataFrame，每行代表一个股票，每列代表一个因子
        market_values (Series): 总市值的 Series，索引为股票代码，值为总市值
        industry_dummies (DataFrame): 行业虚拟变量的 DataFrame，每行代表一个股票，每列代表一个行业

    Returns:
        DataFrame: 市值和行业中性化处理后的因子值的 DataFrame
    """
    # 在 stock 中添加一列表示常数项
    factor_values = sm.add_constant(factor_values)

    # 将市值和行业虚拟变量添加到因子值 DataFrame 中
    factor_values['market_value'] = market_values
    factor_values = pd.concat([factor_values, industry_dummies], axis=1)

    # 执行多元回归，以市值和行业虚拟变量作为自变量，因子值作为因变量
    model = sm.OLS(factor_values.iloc[:, 0], factor_values.iloc[:, 1:])
    result = model.fit()

    # 获取残差项，即市值和行业中性化后的因子值
    residual = result.resid

    return residual




#评价部分
def rank_ic(future_returns, factor_values):
    """
    计算秩相关系数
    Args:
        future_returns (Series): 未来收益的 Series，索引为股票代码，值为未来收益
        factor_values (Series): 因子值的 Series，索引为股票代码，值为因子值

    Returns:
        float: 秩相关系数
    """
    rank_factor_values = rankdata(factor_values)
    rank_future_returns = rankdata(future_returns)
    return np.corrcoef(rank_future_returns, rank_factor_values)[0, 1]

def information_ratio(rank_ic):
    """
    计算信息比率
    Args:
        rank_ic (float): 秩相关系数

    Returns:
        float: 信息比率
    """
    if np.std(rank_ic) == 0:
        return np.nan
    return np.mean(rank_ic) / np.std(rank_ic)

def t_value(rank_ic, n):
    """
    计算 t 值
    Args:
        rank_ic (float): 秩相关系数
        n (int): 样本量

    Returns:
        float: t 值
    """
    if np.std(rank_ic) == 0:
        return np.nan
    return np.mean(rank_ic) / np.std(rank_ic) * np.sqrt(n)

def win_rate(rank_ic):
    """
    计算 RankIC 胜率
    Args:
        rank_ic (Series): 秩相关系数的 Series

    Returns:
        float: RankIC 胜率
    """
    return (rank_ic > 0).sum() / len(rank_ic)

def long_return(factor_values, returns, k):
    """
    计算多头年化收益
    Args:
        factor_values (Series): 因子值的 Series，索引为股票代码，值为因子值
        returns (Series): 收益的 Series，索引为股票代码，值为收益
        k (float): 多头组合的比例（例如前 k%）

    Returns:
        float: 多头年化收益
    """
    threshold = np.percentile(factor_values, 100 - k)
    long_returns = returns[factor_values >= threshold]
    return long_returns.mean()

def short_return(factor_values, returns, k):
    """
    计算空头年化收益
    Args:
        factor_values (Series): 因子值的 Series，索引为股票代码，值为因子值
        returns (Series): 收益的 Series，索引为股票代码，值为收益
        k (float): 空头组合的比例（例如后 k%）

    Returns:
        float: 空头年化收益
    """
    threshold = np.percentile(factor_values, k)
    short_returns = returns[factor_values <= threshold]
    return short_returns.mean()

def long_short_return(factor_values, returns, k):
    """
    计算多空年化收益
    Args:
        factor_values (Series): 因子值的 Series，索引为股票代码，值为因子值
        returns (Series): 收益的 Series，索引为股票代码，值为收益
        k (float): 多头和空头组合的比例（例如前 k% 和后 k%）

    Returns:
        float: 多空年化收益
    """
    long_ret = long_return(factor_values, returns, k)
    short_ret = short_return(factor_values, returns, k)
    return long_ret - short_ret

def long_short_volatility(long_short_returns):
    """
    计算多空波动率
    Args:
        long_short_returns (Series): 多空组合的收益 Series

    Returns:
        float: 多空波动率
    """
    return long_short_returns.std() * np.sqrt(len(long_short_returns))

def long_short_sharpe(long_short_returns):
    """
    计算年化多空夏普比率
    Args:
        long_short_returns (Series): 多空组合的收益 Series

    Returns:
        float: 年化多空夏普比率
    """
    return long_short_returns.mean() / long_short_returns.std() * np.sqrt(len(long_short_returns))


















