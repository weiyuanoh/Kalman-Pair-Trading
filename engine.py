import pandas as pd
import numpy as np 

class TradingEngine:
    def __init__(self, initial_capital=100000.0, commission_bps=0):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.positions = {} # {ticker: {'qty': 0, 'avg_price': 0.0}}
        self.history = []
        self.equity_curve = [] # Track daily portfolio value

    def get_portfolio_value(self, current_prices):
        """Returns Cash + Market Value of Positions"""
        market_value = 0.0
        for ticker, pos in self.positions.items():
            if ticker in current_prices:
                market_value += pos['qty'] * current_prices[ticker]
        return self.cash + market_value

    def execute_order(self, ticker, qty, price, date, action_label):
        """
        Handles Buy/Sell logic including Cash deduction.
        """
        if qty == 0: 
            return

        # Calculate Cost & Commission
        notional = abs(qty * price)
        comm = notional * self.commission_bps 
        
        self.cash -= (qty * price) 
        self.cash -= comm
        
        # Update Position 
        current_pos = self.positions.get(ticker, {'qty': 0, 'avg_price': 0.0})
        curr_qty = current_pos['qty']
        
        # Check direction
        is_same_direction = (curr_qty * qty) > 0 or curr_qty == 0
        
        if is_same_direction:
            # scale in 
            new_total_qty = curr_qty + qty
            old_notional = curr_qty * current_pos['avg_price']
            new_trade_notional = qty * price
        
            new_avg_price = price
            if new_total_qty != 0:
                new_avg_price = (old_notional + new_trade_notional) / new_total_qty
            
            self.positions[ticker] = {'qty': new_total_qty, 'avg_price': new_avg_price}
            
        else:
            # update quantity 
            new_total_qty = curr_qty + qty
            if new_total_qty == 0:
                if ticker in self.positions: del self.positions[ticker]
            else:
                is_flip = (curr_qty * new_total_qty) < 0
                new_avg_price = price if is_flip else current_pos['avg_price']
                self.positions[ticker] = {'qty': new_total_qty, 'avg_price': new_avg_price}

        # log trade
        self.history.append({
            'date': date, 'ticker': ticker, 'action': action_label,
            'qty': qty, 'price': price, 'comm': comm
        })
        
def portfolio_analytics(equity: pd.Series):
    equity_daily = equity.resample('B').last().dropna()
    rets_daily = equity_daily.pct_change().dropna()
    
    # Sharpe 
    if rets_daily.std() != 0:
        sharpe = (rets_daily.mean() / rets_daily.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Vol 
    ann_vol = rets_daily.std() * np.sqrt(252)

    # CAGR 
    start_date = equity_daily.index[0]
    end_date = equity_daily.index[-1]
    years = (end_date - start_date).days / 365.25
    
    if years > 0:
        cagr = (equity_daily.iloc[-1] / equity_daily.iloc[0]) ** (1/years) - 1 
    else:
        cagr = 0.0
    
    # DD
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = dd.min()

    # Max DD duration (in days)
    peak_daily = equity_daily.cummax()
    underwater_daily = equity_daily < peak_daily
    
    dd_duration = (underwater_daily.groupby((underwater_daily != underwater_daily.shift()).cumsum())
                   .cumcount() + 1)
    max_dd_duration = dd_duration[underwater_daily].max() if underwater_daily.any() else 0

    return {
        "Final Equity": float(equity.iloc[-1]),
        "Total Return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "CAGR": float(cagr),
        "Ann Vol": float(ann_vol),
        "Sharpe (rf=0)": float(sharpe),
        "Max Drawdown": float(max_dd),
        "Max DD Duration (days)": int(max_dd_duration),
    }, dd, rets_daily