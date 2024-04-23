# -*- coding: utf-8 -*-

"""
股票池：
上证50、沪深300、中证500、中证800、中证1000等、国证2000、全市场股票（或中证全指）等；
基金池：
这个可以分为几个池子，比如基准池、核心池等；
"""


factor_all = [
    '688399.SH', '300865.SZ', '600755.SH', '002524.SZ', '600319.SH', '300235.SZ', '300836.SZ', '002535.SZ', '603201.SH',
    '688129.SH', '000600.SZ',
]

universe_XGboost = [
    '688399.SH', '300865.SZ', '600755.SH', '002524.SZ', '600319.SH', '300235.SZ', '300836.SZ', '002535.SZ', '603201.SH',
    '688129.SH', '000600.SZ', '002872.SZ', '601857.SH', '300717.SZ', '002652.SZ', '300621.SZ', '300995.SZ', '300639.SZ',
    '002883.SZ', '301036.SZ', '601186.SH', '600525.SH', '600883.SH', '600768.SH', '600243.SH', '300462.SZ', '300626.SZ',
    '300971.SZ', '688251.SH', '600170.SH', '000972.SZ', '688228.SH', '300084.SZ', '601669.SH', '600658.SH', '600089.SH',
    '600710.SH', '601390.SH', '600250.SH', '600438.SH', '600615.SH', '002592.SZ', '300729.SZ', '600180.SH', '600519.SH',
    '300555.SZ', '601898.SH', '600791.SH', '002416.SZ', '601918.SH', '600239.SH', '600822.SH', '603101.SH', '002820.SZ',
    '600674.SH', '601088.SH', '002012.SZ', '600070.SH', '002471.SZ', '000897.SZ', '000669.SZ', '300269.SZ', '600502.SH',
    '600293.SH', '300116.SZ', '600153.SH', '000005.SZ', '000609.SZ', '601919.SH', '002502.SZ', '300140.SZ', '688196.SH'
]


universe_machineLearning = [
    '601919.XSHG', '600104.XSHG', '600196.XSHG', '603259.XSHG', '600309.XSHG', '600436.XSHG', '601995.XSHG',
    '600438.XSHG', '600519.XSHG', '600547.XSHG', '600570.XSHG', '600585.XSHG', '601899.XSHG', '600588.XSHG',
    '600690.XSHG', '600036.XSHG', '600745.XSHG', '601601.XSHG', '600031.XSHG', '600030.XSHG', '603986.XSHG',
    '600028.XSHG', '603288.XSHG', '600050.XSHG', '600276.XSHG', '601888.XSHG', '600000.XSHG', '603501.XSHG',
    '600887.XSHG', '600893.XSHG', '600900.XSHG', '601012.XSHG', '601688.XSHG', '600048.XSHG', '601088.XSHG',
    '600837.XSHG', '601668.XSHG', '601166.XSHG', '601066.XSHG', '600809.XSHG', '601633.XSHG', '601288.XSHG',
    '601318.XSHG', '601628.XSHG', '601336.XSHG', '601398.XSHG', '601138.XSHG', '601857.XSHG', '601211.XSHG',
]

SSE50 = [

]

HS300 = [

]

CS500 = [

]

CS800 = [

]

CS1000 = [

]

CSI2000 = [

]


HS300 = [
    '601318.SH', '002601.SZ', '000002.SZ', '002120.SZ', '601166.SH', '600196.SH', '600000.SH', '002493.SZ',
    '601696.SH', '601162.SH', '600036.SH', '600837.SH', '002008.SZ', '601788.SH', '300759.SZ', '002252.SZ',
    '601229.SH', '002129.SZ', '600763.SH', '603195.SH', '600585.SH', '601898.SH', '601766.SH', '300413.SZ',
    '600958.SH', '603233.SH', '601607.SH', '601009.SH', '601919.SH', '688599.SH', '600030.SH', '601111.SH',
    '600104.SH', '300677.SZ', '002602.SZ', '601398.SH', '600018.SH', '000858.SZ', '688561.SH', '600886.SH',
    '600606.SH', '002179.SZ', '000708.SZ', '601916.SH', '600438.SH', '601006.SH', '000963.SZ', '002044.SZ',
    '000001.SZ', '600383.SH', '002414.SZ', '600183.SH', '601669.SH', '000596.SZ', '002466.SZ', '600588.SH',
    '600570.SH', '000538.SZ', '601186.SH', '601988.SH', '600176.SH', '002001.SZ', '000063.SZ', '600150.SH',
    '600741.SH', '600009.SH', '601818.SH', '300124.SZ', '601990.SH', '000425.SZ', '600584.SH', '603899.SH',
    '600655.SH', '601868.SH', '688981.SH', '002938.SZ', '603392.SH', '601618.SH', '601211.SH', '300142.SZ',
    '002415.SZ', '601108.SH', '300033.SZ', '600109.SH', '600352.SH', '002311.SZ', '300450.SZ', '002024.SZ',
    '300059.SZ', '600900.SH', '000625.SZ', '600015.SH', '002304.SZ', '600690.SH', '600745.SH', '002459.SZ',
    '000333.SZ', '603259.SH', '601601.SH', '603833.SH', '601878.SH', '002049.SZ', '601336.SH', '600926.SH',
    '300433.SZ', '000166.SZ', '300498.SZ', '300888.SZ', '000703.SZ', '600660.SH', '300316.SZ', '300122.SZ',
    '601225.SH', '600519.SH', '002916.SZ', '002812.SZ', '600346.SH', '600079.SH', '300628.SZ', '300595.SZ',
    '688009.SH', '002709.SZ', '300676.SZ', '601155.SH', '002371.SZ', '601985.SH', '600048.SH', '002271.SZ',
    '002241.SZ', '600406.SH', '002410.SZ', '002736.SZ', '000100.SZ', '002157.SZ', '601600.SH', '688111.SH',
    '601877.SH', '601390.SH', '002791.SZ', '601998.SH', '002714.SZ', '000776.SZ', '603369.SH', '600050.SH',
    '603486.SH', '603799.SH', '601319.SH', '300144.SZ', '300558.SZ', '600848.SH', '600919.SH', '000895.SZ',
    '688169.SH', '601668.SH', '600989.SH', '603806.SH', '603288.SH', '000725.SZ', '601698.SH', '000977.SZ',
    '601728.SH', '600436.SH', '002050.SZ', '601360.SH', '601628.SH', '601966.SH', '002032.SZ', '688396.SH',
    '002475.SZ', '600309.SH', '000568.SZ', '300015.SZ', '600600.SH', '603993.SH', '603939.SH', '601012.SH',
    '601901.SH', '601216.SH', '600025.SH', '300408.SZ', '688008.SH', '300999.SZ', '002600.SZ', '002568.SZ',
    '603338.SH', '600918.SH', '688012.SH', '601231.SH', '002607.SZ', '688036.SH', '300347.SZ', '000786.SZ',
    '600115.SH', '603019.SH', '300750.SZ', '600760.SH', '000651.SZ', '001979.SZ', '002555.SZ', '600362.SH',
    '601688.SH', '600016.SH', '600111.SH', '600019.SH', '600332.SH', '002007.SZ', '000069.SZ', '601888.SH',
    '601857.SH', '002460.SZ', '600547.SH', '300601.SZ', '600426.SH', '000783.SZ', '603986.SH', '600031.SH',
    '002841.SZ', '002027.SZ', '601933.SH', '601633.SH', '601838.SH', '601021.SH', '600132.SH', '002236.SZ',
    '000800.SZ', '601100.SH', '601800.SH', '002624.SZ', '600061.SH', '603160.SH', '601236.SH', '600893.SH',
    '002230.SZ', '600905.SH', '603658.SH', '600085.SH', '601865.SH', '603882.SH', '600845.SH', '600161.SH',
    '000301.SZ', '600872.SH', '601816.SH', '000938.SZ', '605499.SH', '300003.SZ', '688126.SH', '000338.SZ',
    '603659.SH', '601995.SH', '300896.SZ', '600143.SH', '600999.SH', '601169.SH', '300274.SZ', '601288.SH',
    '002202.SZ', '601377.SH', '601238.SH', '600795.SH', '601989.SH', '603501.SH', '000661.SZ', '000768.SZ',
    '603087.SH', '601881.SH', '002821.SZ', '600276.SH', '300866.SZ', '688363.SH', '600010.SH', '601939.SH',
    '002352.SZ', '300760.SZ', '002142.SZ', '002064.SZ', '603260.SH', '601799.SH', '601066.SH', '002594.SZ',
    '601899.SH', '000876.SZ'
]


HS300_2009 = [
    '600685.XSHG', '000629.XSHE', '601601.XSHG', '600653.XSHG', '600033.XSHG', '600108.XSHG', '600352.XSHG',
    '600548.XSHG', '000401.XSHE', '600104.XSHG', '601991.XSHG', '000768.XSHE', '000937.XSHE', '000807.XSHE',
    '600030.XSHG', '601727.XSHG', '000538.XSHE', '600350.XSHG', '601958.XSHG', '600690.XSHG', '000927.XSHE',
    '600004.XSHG', '600497.XSHG', '600585.XSHG', '000960.XSHE', '600320.XSHG', '000031.XSHE', '600125.XSHG',
    '000951.XSHE', '000878.XSHE', '600970.XSHG', '600588.XSHG', '600426.XSHG', '601318.XSHG', '600251.XSHG',
    '600028.XSHG', '600085.XSHG', '600096.XSHG', '600550.XSHG', '000623.XSHE', '600547.XSHG',  # , '600357.XSHG'
    '600635.XSHG', '600782.XSHG', '600717.XSHG', '601919.XSHG', '601857.XSHG', '002202.XSHE', '600804.XSHG',
    '601699.XSHG', '600036.XSHG', '600048.XSHG', '601328.XSHG', '600674.XSHG', '000825.XSHE', '000895.XSHE',
    '601998.XSHG', '000680.XSHE', '600837.XSHG', '600795.XSHG', '600616.XSHG', '000968.XSHE', '600188.XSHG',
    '000728.XSHE', '601186.XSHG', '000089.XSHE', '601009.XSHG', '600508.XSHG', '600811.XSHG', '000425.XSHE',
    '000717.XSHE', '000917.XSHE', '600111.XSHG', '000822.XSHE', '000069.XSHE', '600611.XSHG', '002038.XSHE',
    '000959.XSHE', '600016.XSHG', '600153.XSHG', '600331.XSHG', '000652.XSHE', '000488.XSHE', '600395.XSHG',
    '600694.XSHG', '600309.XSHG', '600649.XSHG', '600595.XSHG', '600737.XSHG', '601333.XSHG',   # , '000562.XSHE'
    '600029.XSHG', '600779.XSHG', '600308.XSHG', '601166.XSHG', '600376.XSHG', '000898.XSHE', '000001.XSHE',
    '000793.XSHE', '600143.XSHG', '601600.XSHG', '600256.XSHG', '600839.XSHG', '002128.XSHE', '000932.XSHE',
    '600196.XSHG', '600569.XSHG', '000729.XSHE', '000338.XSHE', '600383.XSHG', '002028.XSHE', '600110.XSHG',
    '601001.XSHG', '601808.XSHG', '600688.XSHG', '600664.XSHG', '600489.XSHG', '600597.XSHG',   # '000046.XSHE',
    '600643.XSHG', '000685.XSHE', '600362.XSHG', '000725.XSHE', '600017.XSHG', '600221.XSHG', '600549.XSHG',
    '600026.XSHG', '600307.XSHG', '600741.XSHG', '000800.XSHE', '600031.XSHG', '000625.XSHE', '600886.XSHG',
    '600266.XSHG', '600428.XSHG', '000423.XSHE', '600177.XSHG', '000858.XSHE', '600118.XSHG', '000686.XSHE',
    '600770.XSHG', '600516.XSHG', '600663.XSHG', '601866.XSHG', '600874.XSHG', '000983.XSHE',   # '600001.XSHG',
    '600812.XSHG', '000778.XSHE', '000002.XSHE', '601939.XSHG', '000157.XSHE', '600600.XSHG',   # '000667.XSHE',
    '000933.XSHE', '600089.XSHG', '600835.XSHG', '002269.XSHE', '600269.XSHG', '600900.XSHG', '601169.XSHG',
    '600316.XSHG', '002122.XSHE', '600018.XSHG', '600655.XSHG', '600006.XSHG', '600019.XSHG', '600169.XSHG',
    '600008.XSHG', '600418.XSHG', '600176.XSHG', '600380.XSHG', '002142.XSHE', '002001.XSHE', '600638.XSHG',
    '600881.XSHG', '600000.XSHG', '600875.XSHG', '002244.XSHE', '000651.XSHE', '601390.XSHG',   # '600102.XSHG',
    '600236.XSHG', '000900.XSHE', '000021.XSHE', '600601.XSHG', '000568.XSHE', '600158.XSHG',   # '600068.XSHG',
    '600027.XSHG', '600015.XSHG', '600219.XSHG', '600037.XSHG', '002024.XSHE', '600528.XSHG', '601988.XSHG',
    '601899.XSHG', '601005.XSHG', '600123.XSHG', '000559.XSHE', '600820.XSHG', '601398.XSHG',   # '600005.XSHG',
    '000718.XSHE', '600583.XSHG', '600675.XSHG', '600100.XSHG', '600718.XSHG', '600010.XSHG', '601766.XSHG',
    '000528.XSHE', '000009.XSHE', '000543.XSHE', '000100.XSHE', '600325.XSHG', '600660.XSHG',   # '600432.XSHG',
    '600879.XSHG', '601666.XSHG', '002242.XSHE', '000027.XSHE', '600050.XSHG', '600022.XSHG',   # '600270.XSHG',
    '600598.XSHG', '600276.XSHG', '000709.XSHE', '601628.XSHG', '600748.XSHG', '600518.XSHG', '600642.XSHG',
    '600210.XSHG', '000969.XSHE', '000612.XSHE', '000039.XSHE', '000783.XSHE', '000063.XSHE', '002155.XSHE',
    '600596.XSHG', '002194.XSHE', '000061.XSHE', '600816.XSHG', '600117.XSHG',   # '600087.XSHG', '600832.XSHG',
    '600183.XSHG', '600058.XSHG', '600170.XSHG', '600216.XSHG', '600151.XSHG', '601898.XSHG', '600809.XSHG',
    '600415.XSHG', '000912.XSHE', '000758.XSHE', '000792.XSHE', '000422.XSHE', '600895.XSHG',   # '600631.XSHG',
    '600132.XSHG', '000012.XSHE', '600456.XSHG', '000060.XSHE', '600739.XSHG', '000059.XSHE', '600109.XSHG',
    '600282.XSHG', '600011.XSHG', '601872.XSHG', '000690.XSHE', '000301.XSHE', '600519.XSHG', '600582.XSHG',
    '000999.XSHE', '601088.XSHG', '600009.XSHG', '600271.XSHG', '600808.XSHG', '000839.XSHE',   # '000024.XSHE',
    '600299.XSHG', '601111.XSHG', '600150.XSHG', '600997.XSHG', '601588.XSHG', '600500.XSHG', '601918.XSHG',
    '600220.XSHG', '600066.XSHG', '600639.XSHG', '601168.XSHG', '000876.XSHE', '000630.XSHE',   # '000527.XSHE',
    '601006.XSHG', '000897.XSHE', '600859.XSHG', '600208.XSHG', '600348.XSHG', '000402.XSHE'
]
