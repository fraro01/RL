import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yfinance as yf

class TradingEnv(gym.Env):
    """
    Ambiente di trading per RL. L'agente può Buy, Hold o Sell una singola azione.
    Stato: finestra mobile di variazioni percentuali dei prezzi di chiusura.
    Reward: differenza monetaria nel valore del portafoglio tra step consecutivi.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, ticker, granularita, sliding_window, start_date, end_date=None):
        super().__init__()
        self.ticker = ticker
        self.granularita = granularita
        self.sliding_window = sliding_window
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = 10000.0  # Capitale iniziale

        # Carica i dati storici di chiusura usando yfinance:contentReference[oaicite:6]{index=6}
        # Se end_date è None, yfinance scarica fino ad oggi.
        data = yf.download(
            tickers=self.ticker,
            start=self.start_date,
            end=self.end_date,
            interval=self.granularita,
            progress=False
        )
        if data is None or data.empty or 'Close' not in data:
            raise ValueError("Impossibile scaricare dati per il ticker o intervallo specificato.")
        # Serie dei prezzi di chiusura
        self.prices = data['Close'].values
        # Numero di step disponibile
        self.max_step = len(self.prices)
        if self.max_step < self.sliding_window + 1:
            raise ValueError("Dati insufficienti per la finestra sliding_window richiesta.")

        # Calcola le variazioni percentuali giornaliere (np array di lunghezza = len(prices))
        # pct_change[0] = 0, poi (P[i]-P[i-1])/P[i-1] per i>=1
        pct = np.zeros(self.max_step, dtype=np.float32)
        for i in range(1, self.max_step):
            pct[i] = (self.prices[i] - self.prices[i-1]) / self.prices[i-1]
        self.pct_changes = pct

        # Spazio di azione e osservazione (Gym)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.sliding_window,), dtype=np.float32
        )

        # Stato del portafoglio
        self.cash = None
        self.shares = None
        self.current_step = None  # indice corrente dell'ambiente

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Inizializza portafoglio
        self.cash = self.initial_capital
        self.shares = 0
        # Inizia dallo step = sliding_window
        # (il primo stato osservato utilizza i primi 'sliding_window' valori di pct_change)
        self.current_step = self.sliding_window
        # Valore portafoglio iniziale (da usare come riferimento)
        self.prev_portfolio_value = self.cash
        # Osservazione iniziale
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        """
        Restituisce l'osservazione corrente: array di variazioni percentuali.
        """
        start = self.current_step - self.sliding_window
        end = self.current_step
        obs = self.pct_changes[start:end]
        return obs.astype(np.float32)

    def step(self, action):
        # Controlla validità dell'azione
        assert self.action_space.contains(action), "Azione non valida."

        done = False
        terminated = False
        truncated = False

        # Prezzo corrente di chiusura
        price = self.prices[self.current_step]

        # Valore del portafoglio prima dell'azione
        old_portfolio = self.cash + self.shares * price

        # Esegui l'azione
        if action == 0:  # Buy
            # Compra 1 azione se possibile
            if self.cash >= price:
                self.shares += 1
                self.cash -= price
            # altrimenti si ignora l'azione (equivale a Hold)
        elif action == 2:  # Sell
            # Vendi 1 azione se possibile
            if self.shares > 0:
                self.shares -= 1
                self.cash += price
            # altrimenti Hold

        # Passa al prossimo passo temporale
        self.current_step += 1

        # Calcola reward: differenza monetaria tra portafoglio nuovo e vecchio
        if self.current_step < self.max_step:
            new_price = self.prices[self.current_step]
            new_portfolio = self.cash + self.shares * new_price
            reward = new_portfolio - old_portfolio
        else:
            # Non ci sono più prezzi successivi
            new_portfolio = self.cash + self.shares * price
            reward = new_portfolio - old_portfolio
            terminated = True

        # Check condizioni di terminazione
        # Episodio finisce se esauriti i dati o portafoglio = 0
        if new_portfolio <= 0:
            terminated = True

        # Ottieni la prossima osservazione (se non terminato)
        if not terminated:
            obs = self._get_obs()
        else:
            # Se terminato, restituisce comunque l'ultima osservazione valida
            obs = self._get_obs()

        info = {}  # informazioni addizionali (non usate)
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        Opzionale: mostra stato attuale del portafoglio.
        """
        price = self.prices[self.current_step] if self.current_step < self.max_step else self.prices[-1]
        total_value = self.cash + self.shares * price
        print(f"Step {self.current_step}: Prezzo={price:.2f}, Cash={self.cash:.2f}, "
              f"Azioni={self.shares}, ValoreTot={total_value:.2f}")

# Esempio di utilizzo:
# env = TradingEnv("AAPL", "1d", sliding_window=10, start_date="2020-01-01", end_date="2021-01-01")
# obs, info = env.reset()
# action = env.action_space.sample()  # es. 0=buy,1=hold,2=sell
# obs, reward, done, truncated, info = env.step(action)
