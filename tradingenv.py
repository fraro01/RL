import gymnasium as gym #for inheritance for the environment
from gymnasium import spaces #for defining the action and observation spaces sticking to the Gym API
import numpy as np #for array manipulations and numerical calculations
import yfinance as yf #for downloading historical stock data
import matplotlib.pyplot as plt #for visualizing the data

class TradingEnv(gym.Env): #inheritance from the Gym parent class

    """
    Trading environment for Reinforcement Learning. The agent can Buy, Hold or Sell ONE single stock per action.
    State: Sliding window of percentage variations of the closing prices.
    Reward: Monetary difference in the wallet value (cash + shares value), between two consecutive steps.
    """

    metadata = {'render_modes': ['human']} #how to render the environment, the supported modes

    def __init__(self, ticker, granularity, sliding_window, start_date, end_date=None, initial_cash=10000, initial_share=0): #inputs for defining the environment
        super().__init__() #inheritance from the gym.Env parent class, to initialize the environment
        self.ticker = ticker #stock ticker (e.g.: "AAPL")
        self.granularity = granularity #data frequency (es. "1d" for daily data)
        self.sliding_window = sliding_window #number of past data to include in the state
        self.start_date = start_date #initial date for data collection (e.g.: "2020-01-01")
        self.end_date = end_date #final date for data collection (e.g.: "2021-01-01"); if None, it will load data until available today
        self.initial_cash = initial_cash  #inital available capital for trading
        self.initial_share = initial_share #initial number of shares held, usually 0

        #load the closing prices using yfinance:contentReference[oaicite:6]{index=6}
        #If end_date is None, yfinance loads up untill available data (hopefully today)
        data = yf.download(
            tickers=self.ticker,
            start=self.start_date,
            end=self.end_date,
            interval=self.granularity,
            progress=False #do not show the loading progress bar
        )
        self.plot_data = data.copy() #we keep a copy of the data for visualization purposes, without modifying it with the cleaning operations we do below  

        if data is None or data.empty or 'Close' not in data:
            raise ValueError("It is not possibe to load the data for the selected ticker or specified interval.")
        # We only select the Closing prices
        self.prices = data['Close'].values
        # Number of max steps in the episode (length of available data)
        self.max_step = len(self.prices)
        if self.max_step < self.sliding_window + 1:
            raise ValueError("Insufficient data for the sliding window size requested.")

        # Calculate the percentage variations of prices for all loaded prices  
        # pct_change[0] = 0, then (P[i]-P[i-1])/P[i-1] for i>=1
        pct = np.zeros(self.max_step, dtype=np.float32) #notice that by doing so, we leave the first value = 0 %
        for i in range(1, self.max_step):
            pct[i] = (self.prices[i] - self.prices[i-1]) / self.prices[i-1]
        self.pct_changes = pct

        # Action and Observation space, defined to be copmliant with Gym
        self.action_space = spaces.Discrete(3) #discrete space with only 3 possible actions to take
        self.observation_space = spaces.Box( # Box is a Gym class used for continuous numerical arrays
                                            low=-np.inf, #there is no inferior limit to percentage changes (theoretically), so we set it to -inf
                                            high=np.inf, #same as above
                                            shape=(self.sliding_window,), #the observation is a monodimensional array with length=len(sliding_window)
                                            dtype=np.float32
                                        )

        #CURRENT wallet state variables, to keep track during the simulation, we initalize them to None
        #they are defined here in the __init__ so they can always be taken
        self.cash = None #available cash to buy stocks, varies every time we buy or sell a stock
        self.shares = None #number of shares currently held
        self.current_step = None  #current index of the environment (which price we are looking at)


        
    #initializes the environment for a new episode, resetting all the variables to their initial state
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) #we call the reset of the base class gym.env, by doing so we can even insert the chosen seed.
        #Initialize first wallet state, => we do not belong any share and we have all the initial capital in cash
        self.cash = self.initial_cash
        self.shares = self.initial_share
        # We start from the step = sliding_window
        # (The first observed state uses the first 'sliding_window' values of pct_change, (so the last one, i.e.: the sliding_window-th)!), 
        self.current_step = self.sliding_window
        # Value of initial wallet, to be used as reference for the reward in the first step
        self.prev_portfolio_value = self.cash
        #first observation (state) of the environment, which is the array of percentage changes for the first 'sliding_window' prices
        obs = self._get_obs()
        return obs, {} #the first return value is the initial observation, the second is an empty dictionary for additional info (not used here)

    #this is a helper function to get the current observation (state) of the environment, which is the array of percentage changes for the last 'sliding_window' prices
    def _get_obs(self):
        """
        Gives the current observation (state) of the environment, which is the array of percentage changes for the last 'sliding_window' prices.
        """
        start = self.current_step - self.sliding_window
        end = self.current_step
        obs = self.pct_changes[start:end] #array of percentage changes for the last 'sliding_window' prices, which is the current state of the environment
        return obs.astype(np.float32)

    #CORE METHOD!
    def step(self, action): #it takes the action chosen as input
        #Check validity of the action
        assert self.action_space.contains(action), "Not valid action."

        #initialize termination flags
        done = False #not used anymore from Gym
        terminated = False #finishes when we run out of data or when the portfolio value is zero or negative
        truncated = False #finishes when we reach the maximum number of steps 

        #CURRENT closing price of the stock, useful for calculating the value of the wallet and for buying or selling stocks
        price = self.prices[self.current_step]

        # Value of the wallet BEFORE TAKING THE ACTION, to be used as reference for the reward calculation
                        #available cash + value of the shares currently held (number of shares * current price per share)
        old_portfolio = self.cash + self.shares * price

        # Submit the action
        #NOTICE, ACTION HOLD=1 DOES NOT HAVE THE CODE BECAUSE DOES NOT MODIFY THE STATE, even though it might be chosen from the policy
        if action == 0:  # Buy
            # Buy one share if it is affordable
            if self.cash >= price:
                self.shares += 1
                self.cash -= price
            # OTHERWISE IT IS THE SAME AS IGNORING THE ACTION => ACTION = 1 => HOLD
        elif action == 2:  # Sell
            # Sell one share if we have any
            if self.shares > 0:
                self.shares -= 1
                self.cash += price
            #OTHERWISE IT IS THE SAME AS IGNORING THE ACTION => ACTION = 1 => HOLD

        # Skip to the next temporal state
        #next observation and reward will be calculated based on the new state of the wallet and the next price in the data
        self.current_step += 1

        # Calculate the reward as the monetary difference between the old and the new wallet
        #NOTICE THAT THIS TYPE OF REWARD IS CALLED "MONETARY REWARD", AND IT IS DIRECTLY LINKED TO THE REAL TRADING GOAL (MAXIMIZE THE WALLET)
        if self.current_step < self.max_step:
            new_price = self.prices[self.current_step]
            new_portfolio = self.cash + self.shares * new_price
            reward = new_portfolio - old_portfolio
        else:
            #there are not succesive prices anymore
            new_portfolio = self.cash + self.shares * price
            reward = new_portfolio - old_portfolio
            terminated = True #end of the episode

        # Check terminal conditions
        #the episode ends if we run out of data or when the portfolio value is zero or negative (bankruptcy)
        if new_portfolio <= 0:
            terminated = True

        # If the episode is noth ended, obtain the next observation
        if not terminated:
            obs = self._get_obs()
        else:
            # If it is terminated it returns the last observation valid in any case
            obs = self._get_obs()

        info = {}  # additional info, not used here now
        return obs, reward, terminated, truncated, info

    #for showing data
    def show_data(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.plot_data.index, self.plot_data['Close'])
        plt.title(f'{self.ticker } Closing Price')
        plt.xlabel('Time')
        plt.ylabel('Closing Price [$]')
        plt.grid(True)
        plt.show()
        
    #for showing the actions taken in relation with the data
    def show_data_with_actions(self, deeds):
        #initial padding, since in the first sliding_window-th prices we don't take any action
        padding = [1 for _ in range(self.sliding_window)] #we pad them as holding
        #padded actions
        z = padding + deeds
        #dimensions
        plt.figure(figsize=(10, 5))
        #Plot of the closing prices
        plt.plot(self.plot_data.index, self.plot_data['Close'])
        #allign deeds to the first len(z) timestamps
        z = np.array(z) #conversion in Numpy, in order to usee boolean masks
        aligned_index = self.plot_data.index[:len(z)] #take only the first len(z) elements
        aligned_price = self.plot_data['Close'].values[:len(z)] #same as above
        #Masks
        #boleean arrays
        mask_red = (z == 0)
        mask_green = (z == 2)
        #Scatter red (z == 0)
        plt.scatter(
            aligned_index[mask_red], #take only v==0 dates
            aligned_price[mask_red], #take related prices
            color='red',
            label='Buy',
            s=20, #dimensions of the markers
            zorder=0 #sovraposition order
        )
        #Scatter green (z == 2)
        plt.scatter(
            aligned_index[mask_green], #same as before
            aligned_price[mask_green],
            color='green',
            label='Sell',
            s=20, 
            zorder=0
        )
        #rendering
        plt.title(f'{self.ticker } Closing Price with Actions')
        plt.xlabel('Time')
        plt.ylabel('Closing Price [$]')
        plt.legend()
        plt.grid(True)
        plt.show()

    #related to the "metadata" dictionary defined at the top
    def render(self, mode='human'): #visualize the current state value while we are interacting with the environment
        """
        Optional, it outputs the current state of the wallet
        """
        price = self.prices[self.current_step] if self.current_step < self.max_step else self.prices[-1]
        total_value = self.cash + self.shares * price
        print(f"Step {self.current_step}: Price={price:.2f}, Cash={self.cash:.2f}, "
              f"Shares={self.shares}, TotalValue={total_value:.2f}")

""""
example of how to use this environment:
env = TradingEnv(ticker="AAPL", granularity="1d", sliding_window=7, start_date="2020-01-01", end_date="2021-01-01", initial_cash=10000, initial_share=5)
obs, info = env.reset()
action = env.action_space.sample()  # es. 0=buy,1=hold,2=sell
obs, reward, done, truncated, info = env.step(action)
"""
