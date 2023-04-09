# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from xbbg import blp

p = ["1M","3M","6M"]
fx = ['USD','EUR','JPY','GBP', 'CAD','AUD','NZD','CHF','NOK','SEK']

for i in fx:
    for j in fx:
        if i !=j:
            df = blp.bdh(
                tickers=(i+j+'CR CMPN Curncy'), flds=['high', 'low', 'last_price'],
                start_date='1989-01-01', end_date='2023-03-22',
                 )
            file_name = i+j+"CR.csv"
            df.to_csv(file_name)