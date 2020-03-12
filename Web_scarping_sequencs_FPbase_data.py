# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:39:21 2020

@author: kuchenov
"""
#Loading Data
from bs4 import BeautifulSoup as bsoup
import requests as rq
import re
import pandas as pd

#Load data and names of the protei
data = pd.read_csv('D:/Work/Computational/FPbase/20200228_FPbase.csv') 

###proteins with unknown sequence
missing_data=[
    'avGFP480',
    'avGFP509',
    'avGFP510',
    'avGFP514',
    'avGFP523',
    'bfloGFPc1',
    'cpt-sapphire-174-173',
    'Cy115',
    'FPmann',
    'GCaMP6f',
    'GZnP3',
    'iq-EBFP2',
    'iq-mApple',
    'iq-mCerulean3',
    'iq-mKate2',
    'iq-mVenus',
    'jRGECO1a',
    'oxCerulean',
    'shBFP',
    'shBFP-N158SL173I',
    'Xpa']

###bring name s of the data to the same style
missing_data = [x.lower() for x in missing_data]
data['Name'] = data.Name.str.lower()
data['Name'] = data.Name.str.replace('\.','')
data['Name'] = data.Name.str.replace('\(\w+\)','')
data['Name'] = data.Name.str.replace('\+','')
data['Name'] = data.Name.str.replace('\\','')
data['Name'] = data.Name.str.replace('\/','')
data['Name'] = data.Name.str.replace('\(','')
data['Name'] = data.Name.str.replace('\)','')
data['Sequence2'] = 'empty'

def scarpe_seq(data=data,missing_data=missing_data):
    for i in range(len(data)):
        if data.iloc[i,0].strip() not in missing_data:
            fpbase = "https://www.fpbase.org/protein/" + data.iloc[i,0].strip()
            page = rq.get(fpbase)
            soup = bsoup(page.content,'html.parser')
            #sequence = soup.find(id = "aminosequence container")
            sequence = soup.find("div", {"class": "aminosequence container"}).get_text()
            data.Sequence2[i] = sequence
            print(i)
    return data

data = scarpe_seq(data=data,missing_data=missing_data)
data = data.drop(columns = ['Sequence'])
data.to_csv('D:/Work/Computational/FPbase/20200228_FPbase_seq.csv') 