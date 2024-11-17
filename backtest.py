import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from train import sequenceModel, PreprocessedDataset, DeviceDataLoader
from config import SEQUENCE_LENGTH, INPUT_SIZE
from utils import get_device
import seaborn as sns
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')


class Backtester:
    def __init__(self, model, backtest_file, initial_cash=1000000, commission_rate=0.0003, max_stocks=20,
                 buy_threshold=0.01, sell_threshold=-0.01):
        self.metrics = None
        self.portfolio_df = None
        self.signals = None
        self.market_index = None
        self.predictions = None
        self.backtest_file = backtest_file
        self.model = model
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.max_stocks = max_stocks
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.device = get_device()
        self.model = torch.load(model).to(self.device)
        self.model.eval()

    def predict(self):
        device = get_device()
        _dataset = PreprocessedDataset(self.backtest_file, training=False, backtest=True, input_size=INPUT_SIZE)
        loader = DataLoader(_dataset, batch_size=32)
        loader = DeviceDataLoader(loader, device)

        stocks = []
        open_prices = []
        close_prices = []
        datetimes = []
        res = []
        for batch in loader:
            _data, _indices, _mask, _industry, _hour, _open_price, _close_price, _sc, _datetime = batch
            _data = _data.to(self.device)
            _indices = _indices.to(self.device)
            _mask = _mask.to(self.device)
            _industry = _industry.to(self.device)
            _hour = _hour.to(self.device)
            _res = self.model(_data, _indices, _mask, _industry, _hour)
            stocks.extend(_sc)
            open_prices.extend(_open_price.cpu().numpy())
            close_prices.extend(_close_price.cpu().numpy())
            datetimes.extend(_datetime)
            res.extend(_res.detach().cpu().numpy())

        self.predictions = pd.DataFrame({
            'stock_code': stocks,
            'datetime': datetimes,
            'open_price': open_prices,
            'close_price': close_prices,
            'prediction': res
        })
        # 转换 datetime 类型
        self.predictions['datetime'] = pd.to_datetime(self.predictions['datetime'])
        # 排序
        self.predictions = self.predictions.sort_values(by=['datetime', 'stock_code']).reset_index(drop=True)

    def generate_signals(self):
        df = self.predictions.copy()
        df['signal'] = 0  # 初始化信号

        # 生成买入和卖出信号
        df['buy_signal'] = df['prediction'] > self.buy_threshold
        df['sell_signal'] = df['prediction'] < self.sell_threshold

        # 对每个交易日期，按照预测值对股票进行排序
        df['rank'] = df.groupby('datetime')['prediction'].rank(method='first', ascending=False)

        # 只对排名在前 max_stocks 的股票生成买入信号
        df.loc[(df['buy_signal']) & (df['rank'] <= self.max_stocks), 'signal'] = 1
        df.loc[df['sell_signal'], 'signal'] = -1

        df['trade_datetime'] = df['datetime']
        self.signals = df

    def run_backtest(self):
        df = self.signals.copy()
        df = df.sort_values(by=['trade_datetime', 'stock_code']).reset_index(drop=True)

        cash = self.initial_cash
        position = {}  # {stock_code: {...}}
        portfolio_values = []
        dates = sorted(df['trade_datetime'].unique())

        for date in dates:
            day_data = df[df['trade_datetime'] == date]
            total_value = cash

            # 更新持仓市值
            for stock_code in list(position.keys()):
                # 获取股票价格
                stock_price_series = day_data[day_data['stock_code'] == stock_code]['open_price']
                if not stock_price_series.empty:
                    stock_price = float(stock_price_series.values[0])
                    position[stock_code]['last_price'] = stock_price
                else:
                    # 如果当天没有价格数据，使用前一交易时间的价格
                    stock_price = position[stock_code]['last_price']

                total_value += position[stock_code]['shares'] * stock_price

            portfolio_values.append({'datetime': date, 'total_value': total_value})

            # 当前持仓数量
            current_holdings = len(position)

            # 处理交易信号
            for idx, row in day_data.iterrows():
                stock_code = row['stock_code']
                signal = row['signal']
                open_price = float(row['open_price'])

                # 买入信号
                if signal == 1 and stock_code not in position and current_holdings < self.max_stocks:
                    # 计算每只股票的投资金额（等权重分配）
                    allocation = cash / (self.max_stocks - current_holdings)
                    shares_to_buy = int(allocation / (open_price * (1 + self.commission_rate)))
                    if shares_to_buy > 0:
                        cost = shares_to_buy * open_price * (1 + self.commission_rate)
                        if cash >= cost:
                            cash -= cost
                            position[stock_code] = {
                                'shares': shares_to_buy,
                                'cost_price': open_price,
                                'buy_datetime': date,  # 记录买入时间
                                'last_price': open_price
                            }
                            current_holdings += 1
                # 卖出信号
                elif signal == -1 and stock_code in position:
                    buy_date = position[stock_code]['buy_datetime'].date()
                    current_date = date.date()
                    if (current_date - buy_date).days >= 1:
                        # 满足 T+1 规则，可以卖出
                        shares_to_sell = position[stock_code]['shares']
                        proceeds = shares_to_sell * open_price * (1 - self.commission_rate)
                        cash += proceeds
                        del position[stock_code]
                        current_holdings -= 1
        self.portfolio_df = pd.DataFrame(portfolio_values)
        # 确保 total_value 列为浮点数
        self.portfolio_df['total_value'] = self.portfolio_df['total_value'].astype(float)

    def calculate_market_index(self):
        df = self.predictions.copy()
        df = df.sort_values(by=['datetime', 'stock_code'])

        # Pivot表，将股票代码作为列，日期时间作为索引，收盘价作为值
        price_df = df.pivot_table(index='datetime', columns='stock_code', values='close_price')

        # 计算收益率
        returns_df = price_df.pct_change().fillna(0)

        # 计算每个时间点的平均收益率（等权重）
        mean_returns = returns_df.mean(axis=1)

        # 计算累积收益率，构建大盘指数
        cum_returns = (1 + mean_returns).cumprod()

        # 归一化指数，使起始值为1
        cum_returns = cum_returns / cum_returns.iloc[0]

        # 重置索引以便处理 datetime
        cum_returns = cum_returns.reset_index()

        # 不调整 datetime，保留原始的小时级别时间点
        self.market_index = pd.DataFrame({
            'datetime': cum_returns['datetime'],
            'market_index': cum_returns.iloc[:, 1].values  # 注意这里取第二列的值
        })

    def calculate_metrics(self):
        df = self.portfolio_df.copy()
        df['returns'] = df['total_value'].pct_change().fillna(0)
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1

        total_days = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days
        annual_factor = 252 / total_days if total_days > 0 else 1

        annualized_return = df['cumulative_returns'].iloc[-1] * annual_factor
        annualized_volatility = df['returns'].std() * np.sqrt(252)
        risk_free_rate = 0.03
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else np.nan

        roll_max = df['total_value'].cummax()
        drawdown = (df['total_value'] - roll_max) / roll_max
        max_drawdown = drawdown.min()

        self.metrics = {
            'total_return': df['cumulative_returns'].iloc[-1],
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    def plot_results(self):
        df = self.portfolio_df.copy()
        df['normalized_total_value'] = df['total_value'] / df['total_value'].iloc[0]

        # 确保 market_index 已经计算
        if self.market_index is None:
            self.calculate_market_index()

        # 合并市场指数和投资组合数据
        merged_df = pd.merge(df[['datetime', 'normalized_total_value']], self.market_index, on='datetime', how='right')

        # 计算回撤
        roll_max = df['total_value'].cummax()
        drawdown = (df['total_value'] - roll_max) / roll_max

        # 使用 seaborn 美化绘图
        sns.set(style="darkgrid")
        palette = sns.color_palette("Set2")

        # 绘制投资组合价值曲线、市场指数和回撤曲线
        fig, ax1 = plt.subplots(figsize=(16, 9))

        # 绘制投资组合价值曲线和市场指数
        ax1.plot(merged_df['datetime'], merged_df['normalized_total_value'], label='Portfolio', color=palette[0],
                 linewidth=3)
        ax1.plot(merged_df['datetime'], merged_df['market_index'], label='Market Index', color=palette[1], linewidth=3)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Normalized Value', fontsize=12)
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 设置日期格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))

        # 创建第二个坐标轴，绘制回撤曲线
        ax2 = ax1.twinx()
        ax2.plot(df['datetime'], drawdown, label='Drawdown', color=palette[2], linestyle='--', linewidth=3)
        ax2.set_ylabel('Drawdown', fontsize=12)
        ax2.legend(loc='upper right', fontsize=12)

        # 绘制性能指标文本框
        metrics_text = (f"Total Return: {self.metrics['total_return']:.4f}\n"
                        f"Annualized Return: {self.metrics['annualized_return']:.4f}\n"
                        f"Annualized Volatility: {self.metrics['annualized_volatility']:.4f}\n"
                        f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.4f}\n"
                        f"Max Drawdown: {self.metrics['max_drawdown']:.4f}")

        # 在图上添加文本框以显示指标，调整位置以避免遮挡
        plt.gcf().text(0.7, 0.15, metrics_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

        plt.title('Portfolio vs Market Index with Drawdown', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run(self):
        self.predict()
        self.generate_signals()
        self.run_backtest()
        self.calculate_metrics()
        self.calculate_market_index()
        self.plot_results()
        print('Performance Metrics:')
        for key, value in self.metrics.items():
            print(f'{key}: {value:.4f}')


backtester = Backtester('models/model_1731681823.pkl', 'backtestset/backtest_set20241116.csv')
backtester.run()
