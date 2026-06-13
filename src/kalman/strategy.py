import pandas as pd
from kalman.engine import TradingEngine, portfolio_analytics

def make_signal(prm, XY_trade): 
    """
    Takes in only the data that is traded XY_trade.
    
    
    """
    df_trade = XY_trade.copy()
    if prm.trade_by == "posterior_spread":
        traded_col = df_trade['posterior_spread']
    elif prm.trade_by == "innovation":
        traded_col = df_trade['innov']
    else:
        raise ValueError(f"Unsupported trade_by value: {prm.trade_by}")
    
    # zscore calculation
    spread_mean = traded_col.rolling(prm.z_sco_win).mean().shift(1)
    spread_std  = traded_col.rolling(prm.z_sco_win).std().shift(1)
    df_trade['z_score'] = (traded_col - spread_mean) / spread_std

    # trade on next bar 
    df_trade['beta_trade'] = df_trade['beta_hat'].shift(1)
    df_trade['z_trade']    = df_trade['z_score'].shift(1)

    return df_trade

def _close_position(engine, ticker, price, date, label):
    qty = engine.positions.get(ticker, {}).get("qty", 0)
    if qty:
        engine.execute_order(ticker, -qty, price, date, label)


def backtest(XY_trade, coms_bps, prm, indep_var, dep_var, verbose=False):
    """
    Backtest wrapper to initialise trading engine. 
    Takes in XY_trade with z score computed for backtesting 
    """
    engine = TradingEngine(initial_capital=100000, commission_bps=coms_bps)
    current_state = 0 # 0=Flat, 1=Long Spread, -1=Short Spread
    
    for i, (index, row) in enumerate(XY_trade.iterrows()):
        price_x = row[indep_var]
        price_y = row[dep_var]
        date = index
        prices = {indep_var: price_x, dep_var: price_y}

        if i < prm.z_sco_win or pd.isna(row['z_trade']) or pd.isna(row['beta_trade']):
            engine.equity_curve.append({'date': date, 'equity': engine.get_portfolio_value(prices)})
            continue

        z = row['z_trade']
        beta = row['beta_trade']
        
        base_qty = 100
        hedge_qty = int(base_qty * beta)
        
        # Trading Logic
        if current_state == 0 and z < - prm.entry_z:
            # Entry Long Spread: Buy Y, Sell Beta*X
            engine.execute_order(dep_var, base_qty, price_y, date, "BUY Y (Long)")
            engine.execute_order(indep_var, -hedge_qty, price_x, date, "SELL X (Hedge)")
            current_state = 1
            
        elif current_state == 0 and z > prm.entry_z:
            # Entry Short Spread: Sell Y, Buy Beta*X
            engine.execute_order(dep_var, -base_qty, price_y, date, "SELL Y (Short)")
            engine.execute_order(indep_var, hedge_qty, price_x, date, "BUY X (Hedge)")
            current_state = -1
            
        elif current_state == 1 and z >= -prm.exit_z:
            _close_position(engine, dep_var, price_y, date, "CLOSE Y")
            _close_position(engine, indep_var, price_x, date, "CLOSE X")
            current_state = 0
            
        elif current_state == -1 and z <= prm.exit_z:
            _close_position(engine, dep_var, price_y, date, "CLOSE Y")
            _close_position(engine, indep_var, price_x, date, "CLOSE X")
            current_state = 0
            
        engine.equity_curve.append({'date': date, 'equity': engine.get_portfolio_value(prices)})

    if verbose:
        print("Simulation complete.")
    if not engine.equity_curve:
        return pd.DataFrame(columns=["equity"]).rename_axis("date")
    return pd.DataFrame(engine.equity_curve).set_index('date')

