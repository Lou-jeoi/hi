# -*- coding: utf-8 -*-
# 主要功能：文件的保存，结果输出等
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import importlib

try:
    from cal_metric_api import cal_portfolio_metric_dict
except Exception:
    pass


def write_file(data, data_dir, file_name, file_type=None, **kwargs):
    """
        输出各种文件格式的通用函数
    :param data: DataFrame, 需要保存为文件的数据
    :param data_dir: str, 文件保存路径
    :param file_name: str, 文件名
    :param file_type: str, 文件类型
    :param kwargs: 其他输出参数
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)   # 创建多层目录
        # os.mkdir(data_dir)    # 只能创建一层目录
    file_doc = os.path.join(data_dir, file_name)  # 拼接文件路径
    # 如果未指定文件类型，从文件路径中获取后缀名, 如'RVar_RSkew_RKurt.csv'将最后一个切分的csv作为文件类型
    if file_type is None:
        file_type = os.path.splitext(file_doc)[1][1:].lower()
    else:
        pass
    # 根据文件类型调用相应的写入方法
    if file_type == 'csv':
        data.to_csv(file_doc, **kwargs)
        # data.to_csv(file_doc, index=False)
    elif file_type == 'pickle' or file_type == 'pkl':  # 因为pickle文件的后缀名可能是pkl所以设置两个条件
        with open(file_doc, 'wb') as file:
            pickle.dump(data, file)
        # # 用字典型输出
        # with open(context.config['factor_dir'] + context.config['file_name_factor_IO'], "wb") as f:
        #     pickle.dump({'dataTrade': g.dataTrade, 'signals': g.signals, 'dataBenchmark': g.dataBenchmark, 'date_range_src': date_range_src, 'config': context.config}, f)
        # # 方法2：
        # pickle.dump(data, open(file_doc, "wb"))
        # # 方法3：
        # data.to_pickle(file_doc)
    elif file_type == 'json':
        data.to_json(file_doc, orient='table', double_precision=15)
    elif file_type == 'bz2':
        data.to_pickle(file_doc, compression='bz2')
        # data.to_pickle(file_doc, compression='infer')    # 效果与上一行相同
    elif file_type == 'parquet':
        data.to_parquet(file_doc)
    elif file_type == 'xlsx':
        if not data:
            # 如果 data 是空的，直接返回
            return
        with pd.ExcelWriter(file_doc) as writer:
            # 遍历字典中的每个键值对，并将每个DataFrame写入到对应的Sheet中
            for sheet_name, sheet in data.items():
                # 检查 DataFrame 是否为空（没有数据行）
                if sheet.empty:
                    # 如果为空，则仅用列标题创建一个新的 DataFrame
                    sheet = pd.DataFrame(columns=sheet.columns)
                sheet.to_excel(writer, sheet_name=sheet_name)
        # # 简单写法：
        # writer = pd.ExcelWriter(file_doc)
        # data.to_excel(writer, sheet_name=file_name)
        # writer.save()
        # writer.close()
    # 此处可继续添加更多文件类型的处理方式
    else:
        # 如果以上文件类型均不符合且通过扩展名获取的文件类型仍不符合，报错。
        raise ValueError("Unsupported file type.")


def output_result(result, all_result, config, output_dir, timing):
    """
        回测输出结果，将运行完回测的context传入，输出结果到相应的文件中
    :param result: DataFrame, 记录各类配置和cal_metric结果
    :param all_result: dict, 记录回测各种结果
    :param config: dict, 回测配置
    :param output_dir: str, 输出路径
    :param timing: Series, 运行时间表
    :return: result: DataFrame, 记录各类配置和cal_metric结果
             all_result: dict, 记录回测各种结果
    """
    # 将所有绩效评价结果保存到回测结果中
    all_result['all_metric'] = result
    # 添加运行时长信息
    result["timing"] = timing.iloc[-1]["timing"] - timing.iloc[0]["timing"]
    # 重命名行索引为"Portfolio"
    result = result.rename_axis('Portfolio')
    # 重置索引
    result = result.reset_index()
    # 拷贝当前列名
    result_columns = list(result.columns)
    # 添加配置信息到输出结果中
    result = result.assign(**{key: str(value) for key, value in config.items()})
    # 调整result的列的顺序，先config的键，再result的结果列
    result = result.reindex(columns=list(config.keys()) + result_columns)
    # 遍历所有portfolio，输出每个portfolio的结果
    # 控制是否生成运行时间结果
    if config["timing"]:
        all_result["timing"] = timing
    # 控制是否生成run层的结果
    if config["output_run"]:
        write_file(result, output_dir, config["factor_name"] + "_cal_metric output.csv",  index=False)
        for name in ['portfolio_return', 'portfolio_netValue', 'instrument', 'date_range', 'frequency_interval', 'select_interval', 'buy_interval', 'sell_interval']:
            write_file(all_result[name].reset_index(), output_dir, config["factor_name"] + "_" + name + ".csv", index=False)
        if config["timing"]:
            write_file(timing, output_dir, config["factor_name"] + "_timing.csv", index=False)
        for i in range(config["subportfolio_num"] + 1):
            file_dir = os.path.join(output_dir, "portfolio" + str(i))
            # 将original_df输出为pickle和xlsx
            write_file(all_result["portfolio" + str(i)]["attributes_original_df"], file_dir, "portfolio" + str(i) + "_attributes_original_df.pickle")
            write_file(all_result["portfolio" + str(i)]["attributes_original_df"], file_dir, "portfolio" + str(i) + "_attributes_original_df.xlsx")
            if config["attributes_time_df"]:
                # 如果time_df开启，将time_df输出为pickle和xlsx
                write_file(all_result["portfolio" + str(i)]["attributes_time_df"], file_dir, "portfolio" + str(i) + "_attributes_time_df.pickle")
                write_file(all_result["portfolio" + str(i)]["attributes_time_df"], file_dir, "portfolio" + str(i) + "_attributes_time_df.xlsx")
            if config["attributes_security_df"]:
                # 如果security_df开启，将time_df输出为pickle和xlsx
                write_file(all_result["portfolio" + str(i)]["attributes_security_df"], file_dir, "portfolio" + str(i) + "_attributes_security_df.pickle")
                write_file(all_result["portfolio" + str(i)]["attributes_security_df"], file_dir, "portfolio" + str(i) + "_attributes_security_df.xlsx")
            result["timing"] = time.time()

    try:
        # # # # 生成衍生指标
        # # if importlib.util.find_spec('cal_metric_api') or not hasattr(importlib.import_module('cal_metric_api'), 'cal_portfolio_metric_dict'):
        # #     factor_module = importlib.import_module('cal_metric_api')
        # #     all_result = getattr(factor_module, 'cal_portfolio_metric_dict')(all_result, config)
        # from cal_metric_api import cal_portfolio_metric_dict
        all_result = cal_portfolio_metric_dict(all_result, config)
    except Exception:
        pass

    return result, all_result


# def multirun_output(CONF, result, all_result):
def multirun_output(output_dir='result/', factor_name='no_name', result=None, all_result=None):
    """
        multirun层回测输出结果
    :param CONF: dict, 配置项，用以控制输出路径
    :param result: DataFrame, cal_metric结果
    :param all_result: dict, 原始表存储字典
    """
    # write_file(result, CONF["output_dir"][0], CONF["factor_name"][0]+"_result_all_metric.csv")
    write_file(result, output_dir, factor_name+"_result_all_metric.csv")
    # write_file(result, CONF["output_dir"][0], CONF["factor_name"][0] + "_result_all_metric.pickle")
    write_file(result, output_dir, factor_name + "_result_all_metric.pickle")
    # write_file(all_result, CONF["output_dir"][0], CONF["factor_name"][0] + "_result_all_config.pickle")
    write_file(all_result, output_dir, factor_name + "_result_all_config.pickle")


def remove_file(file_dir, file_name):
    """
        删除文件的通用函数
    :param file_dir: str, 文件保存路径
    :param file_name: str, 文件名
    """
    # 判断该文件是否存在
    if os.path.exists(os.path.join(file_dir, file_name)):
        # 如果存在，删除文件
        os.remove(os.path.join(file_dir, file_name))


if __name__ == '__main__':
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df.to_parquet("test.parquet")
    print("done")
"""
    typeList = ['csv', 'pickle', 'json', 'bz2', 'parquet']
    from dataGet_Func import read_file

    for fileType in typeList:
        write_file(data=read_file(data_dir='data/', file_name='2013_002262_trade_pickle4.bz2'),
                   data_dir='test_data/', file_name='fileTest.' + fileType)
        print(fileType + ' successfully written')
    print("done")
"""
