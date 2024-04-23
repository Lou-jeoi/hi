
import re



def normalize_code(code):
    """
        转换证券标的代码；
        参考：
            将其他形式的股票代码转换为聚宽可用的股票代码形式。
            https://www.joinquant.com/help/api/help#api:%E5%85%B6%E4%BB%96%E5%87%BD%E6%95%B0
    :param code, str: 证券标的代码；
    :return: code, str: 规范后的证券标的代码；
    """
    if code.endswith('.sz') or code.endswith('.SZ') or code.endswith('.sh') or code.endswith('.SH'):
        code = code.replace(".sz", ".XSHE").replace(".SZ", ".XSHE").replace(".sh", ".XSHG").replace(".SH", ".XSHG")
    # 替换sz为XSHE，hs为XSHG
    elif code.startswith('sz') or code.startswith('SZ'):
        code = code[2:] + "." + 'XSHE'
    elif code.startswith('hs') or code.startswith('HS'):
        code = code[2:] + "." + 'XSHG'
    else:
        # 假设其他情况下前两个字符为字母，后面为纯数字
        match = re.match(r'([A-Z]+)(\d+)', code)
        if match:
            # 提取出字母部分和数字部分
            letters, numbers = match.groups()
            # 替换为指定格式
            code = f'{numbers}.{letters.upper()}'
    # 确保结果为数字.字母的形式
    if not (code[:-5].isdigit() and code[-4:].isalpha()):
        # raise ValueError("Invalid stock code format")
        print("Invalid stock code format:", code)

    return code

