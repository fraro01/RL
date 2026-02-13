# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 08:18:44 2026

@author: Francesco
"""

from tradingenv import TradingEnv
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

#import of the data
data = yf.download(
    tickers='AAPL',
    start='2022-01-01',
    end='2025-01-01',
    interval='1d',
    progress=False
    )


#testing list
x = [1,1,2,2,0,0,1,2,1,1,1,1,0,0,2,2,1,1,2,0,1,2,0,1,2,0,1,2,0]
y = [1 for _ in range(3-1)]


#function to display
def show_data_with_actions(data, deeds):

    #dimensions
    plt.figure(figsize=(10, 5))

    #Plot of the closing prices
    plt.plot(data.index, data['Close'])

    #allign deeds to the first len(deeds) timestamps
    deeds = np.array(deeds) #conversion in Numpy, in order to usee boolean masks
    aligned_index = data.index[:len(deeds)] #take only the first len(deeds) elements
    aligned_price = data['Close'].values[:len(deeds)] #same as above

    #Masks
    #boleean arrays
    mask_red = (deeds == 0)
    mask_green = (deeds == 2)

    #Scatter red (deeds == 0)
    plt.scatter(
        aligned_index[mask_red], #take only v==0 dates
        aligned_price[mask_red], #take related prices
        color='red',
        label='Buy',
        s=20, #dimensions of the markers
        zorder=0 #sovraposition order
    )

    #Scatter green (deeds == 2)
    plt.scatter(
        aligned_index[mask_green], #same as before
        aligned_price[mask_green],
        color='green',
        label='Sell',
        s=20, 
        zorder=0
    )

    plt.title('Closing Price with Signals')
    plt.xlabel('Time')
    plt.ylabel('Closing Price [$]')
    plt.legend()
    plt.grid(True)
    plt.show()   