# -*- coding: utf-8 -*-
# 主要功能：文件的读取


import codecs
import datetime
import os

import pickle
import pandas as pd
import numpy as np
import functools
import importlib
import tushare as ts
import yaml
from dateutil import parser
import copy
from output_file_Func import write_file


# tushare pro接口
token = '57572d840866f49a3e0fd74722b51e1f09273eca0c08d4824db72617'
pro = ts.pro_api(token)

from dataGet_DB import parse_colName_dict, query_DB



def dataGet_filter(param_dataSrc_lst):
    #  —————————————————————————————— 获取dataTrade和dataFactorSrc ——————————————————————————————
    # 保存每一个用于trade的dataSrc
    dataTrade_lst = []
    # 保存每一个用于signal的dataSrc
    dataFactorSrc_lst = []
    # 遍历每一个param_dataSrc
    for param_dataSrc in param_dataSrc_lst:     # param_dataSrc_lst已经是list形式，不存在单纯键值对；
        colName_derived = param_dataSrc['colName_derived']
        # 不能对param_dataSrc的参数轻易修改，param_dataSrc是原始配置项的指针，一旦修改，所有调用位置全部影响；
        colName_dataSrc = param_dataSrc['colName_dataSrc']
        # # 判断 colName_dataSrc 中是否包含字典类型等非str类型字段名称；(2024年2月1日已重新允许键值对)
        # if isinstance(colName_dataSrc, dict) or [isinstance(i, str) for i in colName_dataSrc if isinstance(colName_dataSrc, list)].count(False) > 0:
        #     print('禁止在 colName_dataSrc 中使用字典类型，需要重命名的字段放在 colName_field_mapping 中！！！')

        # 保存单个param_dataSrc下每个security的数据
        dataSrc_lst = []
        securitys = param_dataSrc['security'] if param_dataSrc['func_name_dataSrc'] != 'read_file' else None
        param_dataSrc.pop('security')   # security被循环读取了，因此param_dataSrc需要去掉。
        # params = {key: value for key, value in param_dataSrc.items() if key != 'security'}
        # param_dataSrc['security']不是列表, 则将其放入列表
        if type(securitys) != list:
            securitys = [securitys]
        # 遍历每一个security
        for security in securitys:
            # 调用param_dataSrc['func_name_dataSrc']函数获取数据
            dataSrc = eval(param_dataSrc['func_name_dataSrc'])(security=security, **param_dataSrc)
            # dataSrc = dataSrc.reset_index() if not pd.DataFrame(dataSrc).empty else dataSrc     # 如果非空才重置索引，如果为None，就不重置索引了。
            if pd.DataFrame(dataSrc).empty:
                print(f"当前标的获取到的数据为空；标的为{security}！注意检查当前数据获取函数、数据文件、数据库等是否有对应证券标的、字段，格式是否正确！")
            # parse_colName_dict()： 解析 字段映射表 和 colName_dataSrc中的源数据字段、因子字段名称；
            # 'rename_dict'：按证券品种类型划分列名，'all_type'为所有类型通用；'colName_factor'：因子需要的列名汇总；'colName_tb'：源数据表需要提供的列名汇总；'tbName_col_key'：列名的键值对中指定需要的标名汇总；
            # 'rename_dict_set'：重命名字典，形式为{'源数据列名':'因子所需字段名称'},可直接用于重命名源数据字段名到因子所需列名；
            rename_dict, colName_factor, colName_tb, tbName_col_key, rename_dict_set, colName_derived, colName_derived_values = parse_colName_dict(security, colName_dataSrc, param_dataSrc['colName_field_mapping'], colName_derived, param_dataSrc['DB_dir'])
            dataSrc = dataSrc.reset_index() if not pd.DataFrame(dataSrc).empty and dataSrc.index.name in (colName_factor + colName_tb + colName_derived_values) else dataSrc     # 如果非空才重置索引，如果为None，就不重置索引了。


            try:
                # 生成衍生指标
                if importlib.util.find_spec('dataDerivedIndicator') or not hasattr(
                        importlib.import_module('dataDerivedIndicator'), 'get_derivedIndicator'):
                    factor_module = importlib.import_module('dataDerivedIndicator')
                    # from dataDerivedIndicator import get_derivedIndicator
                    # dataSrc = get_derivedIndicator(dataSrc, colName_derived)
                    dataSrc = getattr(factor_module, 'get_derivedIndicator')(dataSrc, colName_derived)
            except Exception:
                pass

            # 将股票代码变为security格式
            # dataSrc['ts_code'] = security if security and 'ts_code' in colName_dataSrc else (dataSrc['ts_code'] if not dataSrc.empty and 'ts_code' in colName_dataSrc else None)
            dataSrc['ts_code'] = security if security and 'ts_code' in colName_factor else (dataSrc['ts_code'] if not dataSrc.empty and 'ts_code' in colName_dataSrc else None)
            # # 将股票代码变为security格式；这种情况需要dataSrc是DataFrame格式；
            # dataSrc = dataSrc.assign(ts_code=lambda x: security) if 'ts_code' not in dataSrc.columns else dataSrc
            # 计算收益率，需要按照证券标的、时间排序后计算？？？
            if param_dataSrc['colName_derived'] and ('ret' in param_dataSrc['colName_derived']) and ('close' in dataSrc.columns) and 'ret' not in dataSrc.columns:
                # dataSrc['ret'] = dataSrc['close'].pct_change() if 'ret' in colName_dataSrc and 'close' in dataSrc.columns else None
                dataSrc['ret'] = dataSrc['close'].pct_change()

            # 使用 query_DB 的数据在数据获取时已经被重命名过了，所以不需要重命名。
            if param_dataSrc['func_name_dataSrc'] != 'query_DB':
                # rename_dict_update = rename_dict_set.copy()
                # # 在重命名的字典中找到适用于当前数据的键值对
                # df_rename_col = list(set(dataSrc.columns) & set(rename_dict_update.keys()))
                # df_rename_dict = {k: rename_dict_update[k] for k in rename_dict_update if k in df_rename_col}
                # # 找到键值对中value是重复的值
                # rename_dict_repeated_lst = [item for item in df_rename_dict.values() if list(df_rename_dict.values()).count(item) > 1]
                # # 找到具有重复值得键值对
                # rename_dict_repeated_dict = {k: v for k, v in df_rename_dict.items() if v in rename_dict_repeated_lst}
                # # 同一张表中出现多个列名重命名为一个因子名称时，只保留第一个重命名组，其它的都删掉，不在该表重命名。
                # rename_dict_update = {k: rename_dict_update[k] for k in rename_dict_update if k not in list(rename_dict_repeated_dict.keys())[1:]} if len(rename_dict_repeated_dict) > 1 else rename_dict_update

                # # 已经被重命名过的
                # rename_finish = list(set(rename_dict_set.values()) & set(dataSrc.columns))
                # rename_dict_set = {key: value for key, value in rename_dict_set.items() if value not in rename_finish}
                # 跨数据库、跨表、跨字段：名称映射、对齐、重命名：使用字段映射表重命名源数据列名
                # dataSrc = dataSrc.rename(columns=param_dataSrc['colName_field_mapping']) if isinstance(dataSrc, pd.DataFrame) else(
                #     dataSrc.rename(param_dataSrc['colName_field_mapping'][dataSrc.name] if dataSrc.name in param_dataSrc['colName_field_mapping'].keys() else dataSrc.name) if isinstance(dataSrc, pd.Series) else dataSrc)
                # dataSrc = dataSrc.rename(columns=rename_dict_update) if isinstance(dataSrc, pd.DataFrame) else(
                #     dataSrc.rename(rename_dict_update[dataSrc.name] if dataSrc.name in rename_dict_update.keys() else dataSrc.name) if isinstance(dataSrc, pd.Series) else dataSrc)
                dataSrc = dataSrc.rename(columns=rename_dict_set) if isinstance(dataSrc, pd.DataFrame) else(
                    dataSrc.rename(rename_dict_set[dataSrc.name] if dataSrc.name in rename_dict_set.keys() else dataSrc.name) if isinstance(dataSrc, pd.Series) else dataSrc)
            # 跨数据库、跨表、跨字段：数据变化映射，如单位变换等（如需，可添加）
            # # 按日期升序排序
            # if 'date' in dataSrc.columns:
            #     dataSrc = dataSrc.sort_values(by='date')


            # 将每个security的数据保存到dataSrc_lst
            dataSrc_lst.append(dataSrc)
        # dataSrc为合并所有security数据后的dataframe
        dataSrc = pd.concat(dataSrc_lst, axis=0)

        #  ——————————————————————————————因子所需数据列名检查（可以修改）——————————————————————————————
        # 去除添加的整个的空列；这里目的是删掉因生成衍生表而创建的空列；实际因子需要的数据是否为空，需因子自己判断和处理；dataGet只负责供数，数据完整、列不缺就可以；
        dataSrc = dataSrc.dropna(axis=1, how='all')
        # dataSrc的列名
        col_name = dataSrc.reset_index().columns if isinstance(dataSrc, pd.DataFrame) else (dataSrc.name if isinstance(dataSrc, pd.Series) else [])
        # # 如果colName_dataSrc为列表, 将其转化为字典;预留重命名功能，用于因子研究；
        # if isinstance(colName_dataSrc, list):
        #     colName_dataSrc = functools.reduce(lambda x, y: dict(x.items() | y.items()),
        #                                        map(lambda item: item if isinstance(item, dict) else {item: item}, colName_dataSrc))
        # 所需列名为空或dataSrc的列名不包含所需列名则报错
        # if not colName_dataSrc or not set(colName_dataSrc.keys()).issubset(set(col_name)):
        if not rename_dict_set or not set(rename_dict_set.values()).issubset(set(col_name)):
            print("该数据源不能满足该因子所需要的数据！")
            return None, None, None, True
        # 根据字典对对源数据对应的列名更改为因子需要的列名；
        # dataSrc = dataSrc.rename(columns=colName_dataSrc)[colName_dataSrc.values()]
        # dataSrc = dataSrc[colName_dataSrc] if isinstance(dataSrc, pd.DataFrame) else dataSrc
        # dataSrc = dataSrc.rename(columns=rename_dict_set)[list(rename_dict_set.values()) + list(param_dataSrc['colName_derived'].keys())]
        # 截取数据，由于前面获取数据时已进行重命名，所以此处只截取数据，不再重命名；由于将一些衍生指标计算需要的原始列也放在dataSrc中，如果再次重命名，就可能产生重复列；
        dataSrc = dataSrc[list(rename_dict_set.values()) + list(param_dataSrc['colName_derived'].keys())]

        # # ——————重命名：另一种低效率写法：——————
        # # 修改 colName_dataSrc 存在键值对的列名；
        # if isinstance(param_dataSrc['colName_dataSrc'], list):
        #     for name in param_dataSrc['colName_dataSrc']:
        #         if isinstance(name, dict):
        #             if not set(name.keys()).issubset(set(col_name)):
        #                 return None, None, True
        #             # 根据键值对对源数据对应的列名更改为因子需要的列名；
        #             dataSrc = dataSrc.rename(columns=name)  # 可能同时存在单值和键值对，需要只对键值对修改
        #             param_dataSrc['colName_dataSrc'].remove(name)
        #             param_dataSrc['colName_dataSrc'] = param_dataSrc['colName_dataSrc'] + list(name.values())
        # elif isinstance(param_dataSrc['colName_dataSrc'], dict):
        #     if not set(param_dataSrc['colName_dataSrc'].keys()).issubset(set(col_name)):
        #         return None, None, True
        #     dataSrc = dataSrc.rename(columns=param_dataSrc['colName_dataSrc'])
        #     param_dataSrc['colName_dataSrc'] = param_dataSrc['colName_dataSrc'].values()    # ？？？
        # # 基于重命名后的列名，重新获取列名
        # col_name = dataSrc.reset_index().columns if isinstance(dataSrc, pd.DataFrame) else (dataSrc.name if isinstance(dataSrc, pd.Series) else [])
        # # dataSrc的列名不包含所需列名或所需列名为空则报错
        # if not param_dataSrc['colName_dataSrc'] or (isinstance(param_dataSrc['colName_dataSrc'], list) and not set(param_dataSrc['colName_dataSrc']).issubset(set(col_name))) or (
        #         isinstance(param_dataSrc['colName_dataSrc'], dict) and not set(param_dataSrc['colName_dataSrc'].keys()).issubset(set(col_name))):
        #     print("该数据源不能满足该因子所需要的数据！")
        #     return None, None, True
        # # ——————重命名：另一种低效率写法：（结束）——————

        # 将日期列变为datetime.datetime(时间戳)格式
        # if ('date' in dataSrc.columns) and (param_dataSrc['func_name_dataSrc'] != 'read_file'):
        # if ('date' in dataSrc.columns) and (not all(isinstance(value, datetime.datetime) for value in dataSrc['date'])):
        if 'date' in dataSrc.columns:
            # 日期格式化
            dataSrc_index = pd.to_datetime(dataSrc['date'])
            # 若以1970开头，则使用parser.parse重新格式化
            # dataSrc['date'] = dataSrc['date'].map(lambda x: parser.parse(str(x))) if dataSrc_index[0].year == 1970 else dataSrc_index
            # dataSrc['date'] = dataSrc['date'].map(lambda x: parser.parse(str(x))) if dataSrc_index.iloc[0].year == 1970 else dataSrc_index
            # dataSrc['date'] = dataSrc['date'].map(lambda x: parser.parse(str(x))) if pd.to_datetime(pd.Series(dataSrc_index).drop_duplicates()[0]).year == 1970 else dataSrc_index
            dataSrc['date'] = dataSrc['date'].map(lambda x: parser.parse(str(x))) if pd.to_datetime(pd.Series(dataSrc_index).drop_duplicates().iloc[0]).year == 1970 else dataSrc_index

            # # 直接使用 parser.parse 格式化
            # dataSrc['date'] = dataSrc['date'].apply(lambda x: parser.parse(str(x)))

        # data_usage_purpose为signal或None, 则将dataSrc添加到dataSignal_lst
        if param_dataSrc['data_usage_purpose'] != 'trade':
            dataFactorSrc_lst.append(dataSrc)
        # data_usage_purpose为trade或None, 则将dataSrc添加到dataTrade_lst
        if param_dataSrc['data_usage_purpose'] != 'signal':
            dataTrade_lst.append(dataSrc)

    # 合并数据；如果是多个标的读取同一个不带标的列的数据，就会存在多次读取导致重复的问题，需要去除重复；
    dataTrade = pd.concat(dataTrade_lst, axis=0).drop_duplicates()  # 要注意去重重复值对结果的影响；
    dataTrade = dataTrade.set_index('date') if 'date' in dataTrade.columns else dataTrade
    # # 转透视表；需要考虑情况，dataTrade为Series、DataFrame、None；'colName_tradePrice'字段可能为空；'colName_tradePrice'（'close'）、'ts_code'是否在因子列里面，对应handle_data交易方式要变，整体改变后是否能兼容所有类型因子？？？
    # # 每个数据数据获取函数'colName_tradePrice'字段内容可能不同，需要在因子部分单独制定；部分因子类型，在因子生成过程中，交易价格基于因子结果动态改变，是否兼容？？？是否兼容并行化框架？？？
    # dataTrade = pd.pivot(dataTrade, values=param_dataSrc['colName_tradePrice'], columns=['ts_code'])
    dataFactorSrc = pd.concat(dataFactorSrc_lst, axis=0).drop_duplicates()
    dataFactorSrc = dataFactorSrc.set_index('date') if 'date' in dataFactorSrc.columns else dataFactorSrc
    # assert dataTrade and dataFactorSrc, "缺失用于交易或者信号生成的数据，检查 config、data_usage_purpose 设置"
    #  —————————————————————————————— 获取dataTrade和dataFactorSrc完毕 ——————————————————————————————

    #  —————————————————————————————— 获取benchmark ——————————————————————————————
    # 提取 benchmark 内容及对应数据
    if not param_dataSrc_lst[0]['benchmark']:
        return dataTrade, dataFactorSrc, None, False
    benchmark_lst = []
    for security in list(param_dataSrc_lst[0]['benchmark'].keys()):
        # 获取security的收盘价数据, 并将其列名改为股票名, 添加到benchmark_lst
        # benchmark_lst.append(get_tushare_daily(data_dir=param_dataSrc_lst[0]['data_dir'], security=security).rename(columns={'close': security})[security])
        benchmark_lst.append(get_tushare_daily(data_dir=param_dataSrc_lst[0]['data_dir'], security=security).rename(columns={param_dataSrc['colName_benchmarkPrice']: security})[security])
    # 按日期索引进行合并
    dataBenchmark = pd.concat(benchmark_lst, axis=1).drop_duplicates()
    dataBenchmark.index = pd.to_datetime(dataBenchmark.index) if not pd.DataFrame(dataBenchmark).empty else None  # 需要确保基准返回的时间是datetime.datetime（时间戳）格式
    # dataBenchmark = dataBenchmark.set_index('date') if 'date' in dataBenchmark.columns else dataBenchmark

    return dataTrade, dataFactorSrc, dataBenchmark, False



def read_file(data_dir, file_name, file_type=None, **kwargs):
    """
        主函数，算法执行流程。
    :param data_dir:
    :param file_name: str, 文件名;
    :param file_type: str, 文件类型，默认为空，从后缀名获取文件类型，也可指定文件类型：
    :param **kwargs, 其他参数传递给特定文件类型的读取方法;
    :return: data, DataFrame: 读取文件为DataFrame.
    """
    # data_dir = os.getcwd()    # os.getcwd()+'/data/'
    file_doc = os.path.join(data_dir, file_name)  # 拼接文件路径
    # file_doc = data_dir + file_name
    # file_doc = r'{}\{}.xlsx'.format(data_dir, file_name)  # 两种都不带有路径的“\”
    # 如果未指定文件类型，从文件路径中获取后缀名, 如'RVar_RSkew_RKurt.csv'将最后一个切分的csv作为文件类型
    if file_type is None:
        file_type = os.path.splitext(file_doc)[1][1:].lower()
    else:
        pass
    # 根据文件类型调用相应的读取方法
    if file_type == 'csv':
        # pandas读取csv，可设定index_col等参数
        # data = pd.read_csv(file_doc, **kwargs)
        data = pd.read_csv(file_doc)
    elif file_type == 'xlsx':
        # pandas 读取 xlsx 文件
        # 读取excel所有sheet，则设置sheet_name=None
        data = pd.read_excel(file_doc, engine='openpyxl')  # , sheet_name=sheet_name
    elif file_type == 'pickle' or file_type == 'pkl':  # 因为pickle文件的后缀名可能是pkl所以设置两个条件
        # pandas读取pickle
        with open(file_doc, 'rb') as file:
            data = pickle.load(file)
        # # 方法二：
        # data = pickle.load(open(file_doc, 'rb'))
        # # 方法3
        # data = pd.read_pickle(file_doc)
    elif file_type == 'bz2':
        # pandas读取pickle，压缩方式bz2
        # data = pd.read_pickle(file_doc, **kwargs)
        data = pd.read_pickle(file_doc)
        # # 方法2：
        # data = pd.read_pickle(file_doc, compression='infer', storage_options=None)

    elif file_type == 'json':
        # pandas读取json，可设置orient和double_precision等参数
        # data = pd.read_json(file_doc, **kwargs)
        data = pd.read_json(file_doc)
        # # 方法二：
        # import json
        # with open(file_doc, 'r') as f:
        #     data = json.load(f)

    elif file_type == 'parquet':
        # pandas 读取parquet文件
        data = pd.read_parquet(file_doc)
    # 此处可继续添加更多文件类型的处理方式
    elif file_type == 'yml' or file_type == 'yaml':
        with codecs.open(file_doc, encoding='utf-8') as f:
            return yaml.safe_load(f)
    # # todo:ys：用numpy读取，待测试
    # elif file_type == 'npz' or file_type == 'npy':
    #     data = np.load(file_doc)
    # elif file_type == 'h5':
    #     # pandas 读取 h5 文件
    #     data = pd.read_hdf(file_doc)
    #
    #     # # 方法二：
    #     # import deepdish as dd
    #     # data = dd.io.load(file_doc)
    # elif file_type == 'mat':
    #     import scipy.io as sio
    #     data = sio.loadmat(file_doc)
    # # todo:根据使用情况来参考调整
    # elif file_type == 'db':
    #     import sqlite3
    #     conn = sqlite3.connect(file_doc)
    #     conn.close()    # 注意使用后要关闭；链接先使用，用后关闭；
    else:
        # 如果以上文件类型均不符合且通过扩展名获取的文件类型仍不符合，报错。
        raise ValueError("Unsupported file type.")

    return data


def get_trade_cal(data_dir="data/", **kwargs):
    """
        获取交易日历
    :param data_dir: str, 数据目录路径，默认为 "data/"
    :return: trade_cal, DataFrame, 交易日历数据，包含日期和当天是否开盘
    """
    # 如果交易日历数据文件不存在，则获取交易日历数据并存储到文件
    if not os.path.exists(os.path.join(data_dir, "trade_cal.pickle")):
        # tushare交易日历接口获取数据
        trade_cal = pro.trade_cal()
        # 按照日期进行升序排序
        trade_cal = trade_cal.sort_values(by='cal_date', ascending=True)
        # 将日期列格式转换为 %Y%m%d 格式的日期，并仅保留日期部分
        trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'], format="%Y%m%d").dt.date
        # 将 'pretrade_date' 列的日期格式转换为 %Y%m%d 格式的日期，并仅保留日期部分
        trade_cal['pretrade_date'] = pd.to_datetime(trade_cal['pretrade_date'], format="%Y%m%d").dt.date
        # 将 trade_cal 数据存储到文件 "trade_cal.pickle" 中
        write_file(data=trade_cal, data_dir=data_dir, file_name="trade_cal.pickle")
    else:
        # 如果交易日历数据文件存在，则从文件中读取数据
        trade_cal = read_file(data_dir, file_name="trade_cal.pickle")

    return trade_cal


def get_stock_list(data_dir, list_status='L', fields=['ts_code', 'symbol', 'name', 'area', 'industry', 'list_date'], **kwargs):
    """
        获取股票列表
    :param data_dir: str, 数据目录路径，默认为 "data/"
    :param list_status: str, 上市状态，L上市 D退市 P暂停上市，默认为'L'
    :param fields: list, 获得的信息列
    :return: DataFrame, 上市股票列表和基本信息
    """
    if not os.path.exists(os.path.join(data_dir, 'stock_list.pickle')):
        # 如果股票列表数据文件不存在，则调取tushare获取股票列表数据并存储到文件
        stock_list = pro.stock_basic(
            exchange='',
            list_status=list_status,
            fields=fields)
        write_file(data=stock_list, data_dir=data_dir, file_name='stock_list.pickle')
    else:
        # 如果股票列表数据文件存在，则从文件中读取数据
        stock_list = read_file(data_dir, file_name='stock_list.pickle')

    return stock_list


def attribute_history(data_dir, current_dt, trade_cal, security, count, fields=['open', 'close', 'high', 'low', 'vol'], **kwargs):
    """
        获取历史数据
    :param security: str, 股票代码
    :param current_dt: datetime.date, 当前日期
    :param count: int, 获取当前节点向前多少天的历史数据
    :param fields: Tuple, 需要获取历史数据的哪些列，默认包括开盘价、收盘价、最高价、最低价、交易量
    :return: DataFrame, 包含历史价格和交易量数据
    """
    # end_date: 需要获取的历史数据的最后一天
    end_date = (current_dt - datetime.timedelta(days=1))
    # datetime.timedelta(days=1): 标准化的1天时间范围
    start_date = trade_cal[((trade_cal['is_open'] == 1)
                            & (trade_cal['cal_date'] <= end_date))][-count:].iloc[0, :]['cal_date']
    # 获取从end_date 向前数count个交易日的回测开始日期.这里有问题就是现在的tushare接口是日期倒序排列的所以需要改动

    return attribute_datarange_history(data_dir=data_dir, security=security, start_date=start_date,
                                       end_date=end_date, fields=fields)

def has_digit(string):
    # 定义一个判断字符串里面是否有数字的函数
    for char in string:
        if char.isdigit():
            return True
    return False


def get_tushare_daily(data_dir, security, save=True, start_date=None, end_date=None, **kwargs):
    """
        调取tushare每日行情接口读取日线数据daily
    :param data_dir: str, 数据目录路径，默认为 "data/"
    :param security: str, 股票代码，需要读取日线数据daily的股票
    :param save: 是否存储数据，默认为True
    :return: tushare_daily, DataFrame, 包含该股票所有可获取的日线数据daily
    """
    index_security = ['000300.SH', '000905.SH', '000016.SH', '399006.SZ']
    security = security.replace(".XSHE", ".SZ").replace(".XSHG", ".SH").replace(".ZICN", ".CI").replace(".WI", ".CI")
    # # 特色的财务数据或者其他数据新建单独的文件夹
    # data_dir = data_dir + 'daily/'
    # 如果数据文件不存在，则根据股票代码或指数代码调取tushare接口，根据不同的类型，判断不同的tushare接口
    if not os.path.exists(os.path.join(data_dir, security.replace(".", "_") + ".pickle")):
        # 如果是指数，通过tushare指数日线数据接口调取
        # 判断股票代码是否为指数
        if is_index(security):
        # if security in index_security:
            tushare_daily = pro.index_daily(ts_code=security, start_date=start_date, end_date=end_date)
        # 如果是美股，通过tushare线数据接口调取
        elif not has_digit(security):
            tushare_daily = pro.us_daily(ts_code=security, start_date=start_date, end_date=end_date)
        # 如果是指数，通过tushare线数据接口调取
        elif security.startswith('CI'):
            tushare_daily = pro.ci_daily(ts_code=security, start_date=start_date, end_date=end_date)
        # 如果是可转债，通过tushare可转债日线数据接口调取
        elif 110000 < int(security.split(".")[0]) < 130000:
            # # 单独读取可转债的部分数据
            # fields = 'ts_code, trade_date, close, cb_over_rate'
            # tushare_daily = pro.cb_daily(ts_code=security, start_date=start_date, end_date=end_date, fields=fields)
            tushare_daily = pro.cb_daily(ts_code=security, start_date=start_date, end_date=end_date)
        # 如果是个股，调用tushare股票日线数据获取
        else:
            # 个股行情
            tushare_daily = pro.daily(ts_code=security, start_date=start_date, end_date=end_date)
            if tushare_daily.empty:
                # 考虑基金
                tushare_daily = pro.fund_daily(ts_code=security, start_date=start_date, end_date=end_date)
            # 个股换手率, 股息率数据
            tushare_basic = pro.daily_basic(ts_code=security, start_date=start_date, end_date=end_date,
                                            fields=['ts_code', 'trade_date', 'turnover_rate', 'dv_ratio'])
            # 个股筹码结构
            tushare_chip = pro.cyq_perf(ts_code=security, start_date=start_date, end_date=end_date)
            # 将其他指标合并的tushare_daily
            tushare_daily = functools.reduce(lambda x, y: pd.merge(x, y, on=['ts_code', 'trade_date'], how="left"),
                                             [tushare_daily, tushare_basic, tushare_chip])

        # 按照 'trade_date' 列的值进行升序排序
        tushare_daily = tushare_daily.sort_values(by='trade_date', ascending=True)
        # 将 'trade_date' 列的日期格式转换为 %Y%m%d 格式的日期，并仅保留日期部分
        tushare_daily['trade_date'] = pd.to_datetime(tushare_daily['trade_date'], format="%Y%m%d")
        # 将 'trade_date' 列设置为数据的索引
        tushare_daily = tushare_daily.set_index('trade_date')
        # 将 tushare_daily 数据存储到文件 "security.pickle" 中
        if save:
            write_file(data=tushare_daily, data_dir=data_dir, file_name=security.replace(".", "_") + ".pickle")
    else:
        # 如果日线数据文件存在，则从文件中读取数据
        tushare_daily = read_file(data_dir=data_dir, file_name=security.replace(".", "_") + ".pickle")

    return tushare_daily



def get_tushare_monthly(data_dir, security, save=True):
    # ys：这个生成的数据文件名称与get_tushare_daily生成的名称一样，需要区分：1、生成到不同的路径下；2、生成名称加上类似“_monthly”标志；
    # 如果数据文件不存在，则根据股票代码或指数代码调取tushare接口，根据不同的类型，判断不同的tushare接口
    if not os.path.exists(os.path.join(data_dir, security + ".pickle")):
        # 如果是指数，通过tushare指数日线数据接口调取
        if security == '000300':
            tushare_monthly = pro.index_monthly(ts_code=security + '.SH')
        # 如果是个股，调用tushare股票日线数据获取
        elif int(security) >= 600000:
            tushare_monthly = pro.monthly(ts_code=security + '.SH')
        elif int(security) < 100000:
            tushare_monthly = pro.monthly(ts_code=security + '.SZ')
        # 按照 'trade_date' 列的值进行升序排序
        tushare_monthly = tushare_monthly.sort_values(by='trade_date', ascending=True)
        # 将 'trade_date' 列的日期格式转换为 %Y%m%d 格式的日期，并仅保留日期部分
        tushare_monthly['trade_date'] = pd.to_datetime(tushare_monthly['trade_date'], format="%Y%m%d").dt.date
        # 将 'trade_date' 列设置为数据的索引
        tushare_monthly = tushare_monthly.set_index('trade_date')
        # 将 tushare_daily 数据存储到文件 "security.pickle" 中
        if save:
            write_file(data=tushare_monthly, data_dir=data_dir, file_name=security + ".pickle")
    else:
        # 如果日线数据文件存在，则从文件中读取数据
        tushare_monthly = read_file(data_dir=data_dir, file_name=security + ".pickle")

    return tushare_monthly


def attribute_datarange_history(data_dir, security, start_date=None, end_date=None, data_frequency='daily',
                                fields=None, save=True, **kwargs):
    """
        获取历史数据区间的具体价格信息
        :param data_dir: str, 数据目录路径，默认为 "data/"
    :param security: str, 股票代码，需要读取区间日线数据的股票
    :param start_date: str，开始日期，输入形式：'20180101'
    :param end_date: str，结束日期，输入形式：'20181231'
    :param data_frequency:
    :param fields: Tuple, 需要获取历史数据的哪些列，默认包括开盘价、收盘价、最高价、最低价、交易量；['open', 'close', 'high', 'low', 'vol']
    :return: DataFrame，区间内特定列的数据
    """
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        # start_date = start_date.strftime("%Y%m%d")        # tushare对日期只接受“YYYYMMDD”
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        # end_date = end_date.strftime("%Y%m%d")        # tushare对日期只接受“YYYYMMDD”
    if data_frequency == 'daily':
        # 调用 get_tushare_daily() 函数获取指定股票的交易数据，然后根据给定的起始日期和结束日期选择相应的数据行
        trade_data = get_tushare_daily(data_dir, security, save=save).loc[start_date:end_date, :]
    elif data_frequency == 'monthly':
        trade_data = get_tushare_monthly(data_dir, security, save=save).loc[start_date:end_date, :]
    if fields:
        return trade_data[fields]

    return trade_data


def get_today_data(data_dir, date, security):
    """
        获取今日价格数据
    :param data_dir: str, 数据目录路径
    :param date: str，日期，输入形式：'20180101'
    :param security: str, 股票代码，需要读取当日数据的股票
    :return: today_data, Series, 包含当日股票数据
    """
    try:
        data_daily = get_tushare_daily(data_dir, security)
        try:
            today_data = data_daily.loc[date, :]
        except KeyError:
            data_daily.index = pd.to_datetime(data_daily.index)
            today_data = data_daily.loc[date, :]
    except KeyError:
        today_data = pd.Series()

    return today_data
    # 可被调用 amount(操作股票的数量
    # 下单函数
    # Todo: https://www.ricequant.com/doc/rqalpha-plus/api/api/order_api.html#order-value


# 装饰器、迭代器，用于同时获取多个security；
# 在对应函数前面加上 @security_list_decorator
def security_list_decorator(func):
    def wrapper(*args, **kwargs):
        # fired = False
        security = kwargs.get('security')
        if isinstance(security, list):
            result = []
            for sec in security:
                kwargs['security'] = sec
                res = func(*args, **kwargs)
                # res, fired = func(*args, **kwargs)
                # if fired:
                #     return None, fired
                result.append(res)
            return pd.concat(result, axis=0)
            # return pd.concat(result, axis=0), fired
        else:
            return func(*args, **kwargs)
    return wrapper


def get_portfolio(func):
    def wrapper(data_dir, current_dt, security):
        if isinstance(security, list):
            prices = pd.Series(name='price')
            for stock in security:
                price = func(data_dir, current_dt, stock)
                prices[stock] = price
            return prices
        else:
            return func(data_dir, current_dt, security)
    return wrapper


@get_portfolio
def get_price(data_dir, current_dt, security, **kwargs):
    today_data = get_today_data(data_dir, current_dt, security)
    # 如果当天数据为空，代表停牌，价格等于0
    if len(today_data) == 0:
        price = 0
    else:
        # 使用开盘价作为股票当前价格
        price = today_data['close']
    return price


@get_portfolio
def get_capacity(data_dir, current_dt, security, period=252):
    """
        获取股票交易容量，计算方法：个股过去半年日均成交额的10%
    :param data_dir: str, 数据目录路径
    :param current_dt: datetime.date, 当前日期
    :param security: str, 股票代码
    :param period: int, 周期天数，默认252
    :return: capacity: float, 当天的股票容量
    """
    # 获取交易日历
    trade_cal = get_trade_cal(data_dir=data_dir)
    # 调用attribute_history获取过去半年的交易数据
    today_data = attribute_history(data_dir, current_dt, trade_cal, security, count=int(period / 2))
    # 计算半年的交易量的平均值
    capacity = today_data['vol'].mean()

    return capacity


def get_tushare_cb_basic_local(data_dir, security, save=False, start_date=None, end_date=None, **kwargs):
    """
        调取tushare每日行情接口读取cb_basic
        :param data_dir: str, 数据目录路径，默认为 "data/"
        :param security: str, 股票代码，需要读取daily_basic的股票
        :param save: 是否存储数据，默认为False
        :return: tushare_daily_basic, DataFrame, 包含该股票所有可获取的cb_basic
        """
    # 特色的财务数据或者其他数据新建单独的文件夹
    data_dir = data_dir + 'cb_basic/'
    security = security.replace(".XSHE", ".SZ").replace(".XSHG", ".SH").replace(".ZICN", ".CI").replace(".WI", ".CI")
    # 提前给出相应的数据col
    fields = 'ts_code, remain_size, issue_rating'
    # 如果数据文件不存在，则根据股票代码或指数代码调取tushare接口，根据不同的类型，判断不同的tushare接口
    if not os.path.exists(os.path.join(data_dir, security.replace(".", "_") + ".pickle")):
        # 通过daily_basic调用tushare读取数据
        tushare_cb_basic = pro.cb_basic(ts_code=security, start_date=start_date, end_date=end_date, fields=fields)

        # 将 tushare_daily_basic 数据存储到文件 "security.pickle" 中
        if save:
            write_file(data=tushare_cb_basic, data_dir=data_dir, file_name=security.replace(".", "_") + ".pickle")
    else:
        # 如果日线数据文件存在，则从文件中读取数据
        tushare_cb_basic = read_file(data_dir=data_dir, file_name=security.replace(".", "_") + ".pickle")
    return tushare_cb_basic


def initialize_read_files(data_dir, security, start_date=None, end_date=None, data_list_daily=None,
                          data_list_quarterly=None, **kwargs):
    """
    调取tushare读取数据
    :param data_dir: str, 数据目录路径，默认为 "data/"
    :param security: str, 股票代码
    :param save: 是否存储数据，默认为True
    :return:  df_daily_security, df_quarterly_security, DataFrame
    """
    df_daily_security = pd.DataFrame()
    df_quarterly_security = pd.DataFrame()
    for data_type in data_list_daily:
        func_name = 'get_tushare_{x}'.format(x=data_type)
        get_tushare_function = eval(func_name)
        temp_df = get_tushare_function(data_dir, security, start_date=start_date, end_date=end_date)
        if df_daily_security.empty:
            df_daily_security = temp_df
        else:
            df_daily_security = pd.merge(df_daily_security, temp_df, on='trade_date', suffixes=('', '_right'))
            df_daily_security = df_daily_security.filter(regex='^(?!.*right).*$')

    for data_type in data_list_quarterly:
        func_name = 'get_tushare_{x}'.format(x=data_type)
        get_tushare_function = eval(func_name)
        temp_df = get_tushare_function(data_dir, security, start_date=start_date, end_date=end_date)
        if df_quarterly_security.empty:
            df_quarterly_security = temp_df
        else:
            df_quarterly_security = pd.merge(df_quarterly_security, temp_df, on='trade_date', suffixes=('', '_right'))
            df_quarterly_security = df_quarterly_security.filter(regex='^(?!.*right).*$')
    return df_daily_security
    # return df_daily_security, df_quarterly_security


def get_tushare_cashflow(data_dir, security, save=True, start_date=None, end_date=None, **kwargs):
    """
    调取tushare每日行情接口读取现金流数据
    :param data_dir: str, 数据目录路径，默认为 "data/"
    :param security: str, 股票代码，需要读取现金流数据的股票
    :param save: 是否存储数据，默认为True
    :return: tushare_cashflow, DataFrame, 包含该股票所有可获取的现金流数据
    """
    # 特色的财务数据或者其他数据新建单独的文件夹
    data_dir = data_dir + 'cashflow/'
    security = security.replace(".XSHE", ".SZ").replace(".XSHG", ".SH").replace(".ZICN", ".CI").replace(".WI", ".CI")
    # 如果数据文件不存在，则根据股票代码或指数代码调取tushare接口，根据不同的类型，判断不同的tushare接口
    if not os.path.exists(os.path.join(data_dir, security.replace(".", "_") + ".pickle")):
        # 通过cashflow调用tushare读取现金流数据
        tushare_cashflow = pro.cashflow(ts_code=security, start_date=start_date, end_date=end_date)
        # 将end_date重命名为trade_date，便于之后的数据处理和合并
        tushare_cashflow = tushare_cashflow.rename(columns={'end_date': 'trade_date'})
        # 按照 'trade_date' 列的值进行升序排序
        tushare_cashflow = tushare_cashflow.sort_values(by='trade_date', ascending=True)
        # 将 'trade_date' 列的日期格式转换为 %Y%m%d 格式的日期，并仅保留日期部分
        tushare_cashflow['trade_date'] = pd.to_datetime(tushare_cashflow['trade_date'], format="%Y%m%d")
        # 将 tushare_cashflow 数据存储到文件 "security.pickle" 中
        if save:
            write_file(data=tushare_cashflow, data_dir=data_dir, file_name=security.replace(".", "_") + ".pickle")
    else:
        # 如果日线数据文件存在，则从文件中读取数据
        tushare_cashflow = read_file(data_dir=data_dir, file_name=security.replace(".", "_") + ".pickle")
    return tushare_cashflow


def get_tushare_daily_basic(data_dir, security, save=True, start_date=None, end_date=None, **kwargs):
    """
    调取tushare每日行情接口读取daily_basic
    :param data_dir: str, 数据目录路径，默认为 "data/"
    :param security: str, 股票代码，需要读取daily_basic的股票
    :param save: 是否存储数据，默认为True
    :return: tushare_daily_basic, DataFrame, 包含该股票所有可获取的daily_basic
    """
    # 特色的财务数据或者其他数据新建单独的文件夹
    data_dir = data_dir + 'daily_basic/'
    security = security.replace(".XSHE", ".SZ").replace(".XSHG", ".SH").replace(".ZICN", ".CI").replace(".WI", ".CI")
    # 如果数据文件不存在，则根据股票代码或指数代码调取tushare接口，根据不同的类型，判断不同的tushare接口
    if not os.path.exists(os.path.join(data_dir, security.replace(".", "_") + ".pickle")):
        # 通过daily_basic调用tushare读取数据
        tushare_daily_basic = pro.daily_basic(ts_code=security, start_date=start_date, end_date=end_date)
        # 按照 'trade_date' 列的值进行升序排序
        tushare_daily_basic = tushare_daily_basic.sort_values(by='trade_date', ascending=True)
        # 将 'trade_date' 列的日期格式转换为 %Y%m%d 格式的日期，并仅保留日期部分
        tushare_daily_basic['trade_date'] = pd.to_datetime(tushare_daily_basic['trade_date'], format="%Y%m%d")
        # 将 tushare_daily_basic 数据存储到文件 "security.pickle" 中
        if save:
            write_file(data=tushare_daily_basic, data_dir=data_dir, file_name=security.replace(".", "_") + ".pickle")
    else:
        # 如果日线数据文件存在，则从文件中读取数据
        tushare_daily_basic = read_file(data_dir=data_dir, file_name=security.replace(".", "_") + ".pickle")
    return tushare_daily_basic


def get_tushare_fina_indicator(data_dir, security, save=True, start_date=None, end_date=None, **kwargs):
    """
    调取tushare每日行情接口读取财务数据
    :param data_dir: str, 数据目录路径，默认为 "data/"
    :param security: str, 股票代码，需要读取财务数据的股票
    :param save: 是否存储数据，默认为True
    :return: tushare_fina_indicator, DataFrame, 包含该股票所有可获取的财务数据
    """
    # 特色的财务数据或者其他数据新建单独的文件夹
    data_dir = data_dir + 'fina_indicator/'
    security = security.replace(".XSHE", ".SZ").replace(".XSHG", ".SH").replace(".ZICN", ".CI").replace(".WI", ".CI")
    # 如果数据文件不存在，则根据股票代码或指数代码调取tushare接口，根据不同的类型，判断不同的tushare接口
    if not os.path.exists(os.path.join(data_dir, security.replace(".", "_") + ".pickle")):
        # 通过fina_indicator调用tushare读取财务数据
        tushare_fina_indicator = pro.fina_indicator(ts_code=security, start_date=start_date, end_date=end_date)
        # 将end_date重命名为trade_date，便于之后的数据处理和合并
        tushare_fina_indicator = tushare_fina_indicator.rename(columns={'end_date': 'trade_date'})
        # 按照 'trade_date' 列的值进行升序排序
        tushare_fina_indicator = tushare_fina_indicator.sort_values(by='trade_date', ascending=True)
        # 将 'trade_date' 列的日期格式转换为 %Y%m%d 格式的日期，并仅保留日期部分
        tushare_fina_indicator['trade_date'] = pd.to_datetime(tushare_fina_indicator['trade_date'], format="%Y%m%d")
        # 将 tushare_fina_indicator 数据存储到文件 "security.pickle" 中
        if save:
            write_file(data=tushare_fina_indicator, data_dir=data_dir, file_name=security.replace(".", "_") + ".pickle")
    else:
        # 如果日线数据文件存在，则从文件中读取数据
        tushare_fina_indicator = read_file(data_dir=data_dir, file_name=security.replace(".", "_") + ".pickle")
    return tushare_fina_indicator


def get_tushare_income(data_dir, security, save=True, start_date=None, end_date=None, **kwargs):
    """
    调取tushare利润表接口读取利润表
    :param data_dir: str, 数据目录路径，默认为 "data/"
    :param security: str, 股票代码，需要读取利润表的股票
    :param save: 是否存储数据，默认为True
    :return: tushare_fina_indicator, DataFrame, 包含该股票所有可获取的利润表
    """
    # 特色的财务数据或者其他数据新建单独的文件夹
    data_dir = data_dir + 'income/'
    security = security.replace(".XSHE", ".SZ").replace(".XSHG", ".SH").replace(".ZICN", ".CI").replace(".WI", ".CI")
    # 如果数据文件不存在，则根据股票代码或指数代码调取tushare接口，根据不同的类型，判断不同的tushare接口
    if not os.path.exists(os.path.join(data_dir, security.replace(".", "_") + ".pickle")):
        # 通过fina_indicator调用tushare读取财务数据
        tushare_income = pro.income(ts_code=security, start_date=start_date, end_date=end_date)
        # 将end_date重命名为trade_date，便于之后的数据处理和合并
        tushare_income = tushare_income.rename(columns={'end_date': 'trade_date'})
        # 按照 'trade_date' 列的值进行升序排序
        tushare_income = tushare_income.sort_values(by='trade_date', ascending=True)
        # 将 'trade_date' 列的日期格式转换为 %Y%m%d 格式的日期，并仅保留日期部分
        tushare_income['trade_date'] = pd.to_datetime(tushare_income['trade_date'], format="%Y%m%d")
        # 将 tushare_income 数据存储到文件 "security.pickle" 中
        if save:
            write_file(data=tushare_income, data_dir=data_dir, file_name=security.replace(".", "_") + ".pickle")
    else:
        # 如果日线数据文件存在，则从文件中读取数据
        tushare_income = read_file(data_dir=data_dir, file_name=security.replace(".", "_") + ".pickle")
    return tushare_income


def is_index(ts_code):  # 判断股票代码是否属于指数代码
    security = copy.deepcopy(ts_code)
    security = security.replace(".XSHE", ".SZ").replace(".XSHG", ".SH")
    if (int(security.split('.')[0]) > 800000) or (
            security.split('.')[1] == 'SH' and int(security.split('.')[0]) < 600000) \
            or (security.split('.')[1] == 'SZ' and int(security.split('.')[0]) > 399000):
        return True
    else:
        return False


def get_us_tradecal(data_dir="data/", **kwargs):
    """
    获取交易日历
    :param data_dir: str, 数据目录路径，默认为 "data/"
    :return: trade_cal, DataFrame, 交易日历数据，包含日期和当天是否开盘
    """
    # 如果交易日历数据文件不存在，则获取交易日历数据并存储到文件
    if not os.path.exists(os.path.join(data_dir, "us_trade_cal.pickle")):
        # tushare交易日历接口获取数据
        us_tradecal = pro.us_tradecal()
        # 按照日期进行升序排序
        us_tradecal = us_tradecal.sort_values(by='cal_date', ascending=True)
        # 将日期列格式转换为 %Y%m%d 格式的日期，并仅保留日期部分
        us_tradecal['cal_date'] = pd.to_datetime(us_tradecal['cal_date'], format="%Y%m%d").dt.date
        # 将 'pretrade_date' 列的日期格式转换为 %Y%m%d 格式的日期，并仅保留日期部分
        us_tradecal['pretrade_date'] = pd.to_datetime(us_tradecal['pretrade_date'], format="%Y%m%d").dt.date
        # 将 trade_cal 数据存储到文件 "trade_cal.pickle" 中
        write_file(data=us_tradecal, data_dir=data_dir, file_name="us_trade_cal.pickle")
    else:
        # 如果交易日历数据文件存在，则从文件中读取数据
        us_tradecal = read_file(data_dir, file_name="us_trade_cal.pickle")

    return us_tradecal


def get_basic(data_dir, security, start_date, end_date, fields='ts_code,trade_date,turnover_rate,volume_ratio,pe,pb, '
                                                               'total_mv, circ_mv',
              save=True, **kwargs):
    security = security.replace(".XSHE", ".SZ").replace(".XSHG", ".SH")
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    start_date = start_date.strftime("%Y%m%d")
    end_date = end_date.strftime("%Y%m%d")
    # 如果数据文件不存在，则根据股票代码或指数代码调取tushare接口，根据不同的类型，判断不同的tushare接口
    if not os.path.exists(os.path.join(data_dir, security.replace(".", "_") + "_basic.pickle")):
        tushare_basic = pro.daily_basic(ts_code=security, start_date=start_date, end_date=end_date, fields=fields)
        # 按照 'trade_date' 列的值进行升序排序
        tushare_basic = tushare_basic.sort_values(by='trade_date', ascending=True)
        # 将 'trade_date' 列的日期格式转换为 %Y%m%d 格式的日期，并仅保留日期部分
        tushare_basic['trade_date'] = pd.to_datetime(tushare_basic['trade_date'], format="%Y%m%d")
        # 将 tushare_daily 数据存储到文件 "security.pickle" 中
        if save:
            write_file(data=tushare_basic, data_dir=data_dir, file_name=security.replace(".", "_") + "_basic.pickle")
    else:
        # 如果日线数据文件存在，则从文件中读取数据
        tushare_basic = read_file(data_dir=data_dir, file_name=security.replace(".", "_") + "_basic.pickle")

    return tushare_basic


def get_industry_price(data_dir, security, start_date, end_date, data_frequency='daily',
                       fields=None, save=True,
                       **kwargs):
    date = datetime.datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
    stock_industry = get_industry(security, date)
    if fields is None:
        fields = ['ts_code', 'trade_date', 'open', 'close', 'high', 'low', 'vol', 'change']
    industry = attribute_datarange_history(data_dir, stock_industry, start_date, end_date, data_frequency, fields)
    industry = industry.rename(columns={'ts_code': 'index_code'})  # 重命名
    industry['ts_code'] = security

    return industry


def get_industry(ts_code, trade_date, **kwargs):  # 获取股票在时间范围内的所属行业

    df = pro.index_member(ts_code=ts_code)  # 获取股票的历史所属行业信息
    df = df.sort_values(by='in_date', ascending=False)  # 将信息按照加入时间降序排序
    df = df.reset_index()
    flag = False
    for i in range(len(df)):
        if trade_date > df.loc[df.index[i], 'in_date']:  # 判断当前交易日的股票所属行业
            ind_code = df.loc[df.index[i], 'index_code']  # 找到对应的所属行业指数
            flag = True
            break  # 退出循环
    if not flag:  # 如果没有找到
        ind_code = df.loc[df.index[0], 'index_code']  # 将最新的所属行业作为股票所属行业

    return ind_code


def dataGet_filter_multi(data_dir, benchmark, param_dataSrc, **kwargs):
    signal = {}
    trade = {}
    fired = False
    benchmark_keys = list(benchmark.keys())
    benchmark_data = []
    for key in benchmark_keys:
        benchmark_data.append(attribute_datarange_history(data_dir=data_dir, security=key))
    benchmark_data = pd.concat(benchmark_data)
    if isinstance(param_dataSrc, list):
        for param in param_dataSrc:
            if not (param['data_usage_purpose'] == 'trade'):
                signal[param['func_name_dataSrc']], fired = dataGet_filter(**param)
            if not (param['data_usage_purpose'] == 'signal'):
                trade[param['func_name_dataSrc']], fired = dataGet_filter(**param)
            if fired:
                return None, None, None, fired
        signal = pd.concat(list(signal.values()))
        trade = pd.concat(list(trade.values()))
    else:
        if not (param_dataSrc['data_usage_purpose'] == 'trade'):
            signal[param_dataSrc['func_name_dataSrc']], fired = dataGet_filter(**param_dataSrc)
        if not (param_dataSrc['data_usage_purpose'] == 'signal'):
            trade[param_dataSrc['func_name_dataSrc']], fired = dataGet_filter(**param_dataSrc)
        if fired:
            return None, None, None, fired

    return benchmark_data, trade, signal, fired


def get_index_component(index_code, start_date, end_date):  # 可获取基本指数和行业指数的成分股
    df = pro.index_weight(index_code=index_code, start_date=start_date, end_date=end_date)  # 获取回测周期内股票的历史成分股
    return df


if __name__ == '__main__':
    df = attribute_datarange_history(data_dir="data/", security='000001', start_date="2015-01-01", end_date="2021-12-30",
                                   data_frequency='monthly', save=False)
    print("done")
