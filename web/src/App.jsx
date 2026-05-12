import { useEffect, useMemo, useRef, useState } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';
import * as echarts from 'echarts';

const ETF_NAMES = {
  '159819': '人工智能ETF',
  '512480': '半导体ETF',
  '159770': '机器人ETF',
  '516160': '新能源ETF',
  '561560': '电力ETF',
  '563530': '卫星ETF',
  '515100': '红利低波100ETF',
  '512800': '银行ETF',
};

const DATA_PATHS = {
  signal: '/output/stable_signal/latest_signal.json',
  performance: '/output/no_hotspot_backtest/performance.csv',
  equity: '/output/no_hotspot_backtest/equity_curve.csv',
  weekly: '/output/no_hotspot_backtest/weekly_signals.csv',
  trades: '/output/no_hotspot_backtest/trades.csv',
  news: '/output/stable_signal/news_candidates.csv',
  validation: '/output/no_hotspot_backtest/validation_report.json',
};

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = '';
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];

    if (ch === '"' && quoted && next === '"') {
      cell += '"';
      i += 1;
    } else if (ch === '"') {
      quoted = !quoted;
    } else if (ch === ',' && !quoted) {
      row.push(cell);
      cell = '';
    } else if ((ch === '\n' || ch === '\r') && !quoted) {
      if (ch === '\r' && next === '\n') i += 1;
      row.push(cell);
      if (row.some((value) => value !== '')) rows.push(row);
      row = [];
      cell = '';
    } else {
      cell += ch;
    }
  }

  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }

  const [headers, ...lines] = rows;
  if (!headers) return [];

  return lines.map((line) =>
    Object.fromEntries(
      headers.map((header, index) => {
        const raw = line[index] ?? '';
        const number = Number(raw);
        return [header, raw !== '' && Number.isFinite(number) ? number : raw];
      }),
    ),
  );
}

async function fetchText(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${path} ${response.status}`);
  return response.text();
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${path} ${response.status}`);
  return response.json();
}

function fmtPct(value, digits = 2) {
  if (value === '' || value === undefined || value === null || Number.isNaN(Number(value))) return '--';
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function fmtNum(value, digits = 2) {
  if (value === '' || value === undefined || value === null || Number.isNaN(Number(value))) return '--';
  return Number(value).toFixed(digits);
}

function pctClass(value) {
  if (!Number.isFinite(Number(value))) return '';
  return Number(value) >= 0 ? 'positive' : 'negative';
}

function latest(rows) {
  return rows.length ? rows[rows.length - 1] : null;
}

function calcDrawdown(equityRows) {
  let high = 0;
  return equityRows.map((row) => {
    const value = Number(row.strategy);
    high = Math.max(high || value, value);
    return {
      time: row.date,
      value: high ? ((value / high) - 1) * 100 : 0,
    };
  });
}

function MetricCard({ label, value, tone, hint }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong className={tone}>{value}</strong>
      {hint ? <em>{hint}</em> : null}
    </div>
  );
}

function LoadingPanel({ error }) {
  return (
    <main className="app-shell">
      <section className="empty-state">
        <h1>A 股 ETF 策略面板</h1>
        <p>{error ? `数据加载失败：${error}` : '正在加载策略输出数据...'}</p>
        <code>scripts/run_web_dashboard.sh 2026-05-12</code>
      </section>
    </main>
  );
}

function TradingChart({ equityRows }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || equityRows.length === 0) return undefined;

    const chart = createChart(containerRef.current, {
      autoSize: true,
      layout: {
        background: { color: '#0d1117' },
        textColor: '#aeb7c4',
        fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      },
      grid: {
        vertLines: { color: 'rgba(148, 163, 184, 0.12)' },
        horzLines: { color: 'rgba(148, 163, 184, 0.12)' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: '#6b7280', labelBackgroundColor: '#1f2937' },
        horzLine: { color: '#6b7280', labelBackgroundColor: '#1f2937' },
      },
      rightPriceScale: {
        borderColor: 'rgba(148, 163, 184, 0.25)',
        scaleMargins: { top: 0.08, bottom: 0.18 },
      },
      timeScale: {
        borderColor: 'rgba(148, 163, 184, 0.25)',
        timeVisible: false,
      },
      localization: {
        priceFormatter: (price) => price.toFixed(3),
      },
    });

    const strategy = chart.addAreaSeries({
      title: '策略净值',
      lineColor: '#22c55e',
      topColor: 'rgba(34, 197, 94, 0.32)',
      bottomColor: 'rgba(34, 197, 94, 0.02)',
      lineWidth: 2,
      priceLineVisible: false,
    });
    const benchmark = chart.addLineSeries({
      title: '沪深300',
      color: '#60a5fa',
      lineWidth: 1,
      priceLineVisible: false,
    });
    const defensive = chart.addLineSeries({
      title: '防守基准',
      color: '#f59e0b',
      lineWidth: 1,
      priceLineVisible: false,
    });

    strategy.setData(equityRows.map((row) => ({ time: row.date, value: Number(row.strategy) })));
    benchmark.setData(equityRows.map((row) => ({ time: row.date, value: Number(row.benchmark_hs300) })));
    defensive.setData(equityRows.map((row) => ({ time: row.date, value: Number(row.defensive_static) })));
    chart.timeScale().fitContent();

    return () => chart.remove();
  }, [equityRows]);

  return <div className="trading-chart" ref={containerRef} />;
}

function EChart({ option, className }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return undefined;
    const chart = echarts.init(containerRef.current, null, { renderer: 'canvas' });
    chart.setOption(option);

    const resize = () => chart.resize();
    window.addEventListener('resize', resize);
    return () => {
      window.removeEventListener('resize', resize);
      chart.dispose();
    };
  }, [option]);

  return <div className={className} ref={containerRef} />;
}

function StatusPill({ bucket }) {
  const label = bucket === 'offense' ? '进攻' : bucket === 'defense' ? '防守' : '均衡';
  return <span className={`pill ${bucket || 'balanced'}`}>{label}</span>;
}

function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [signal, performanceText, equityText, weeklyText, tradesText, newsText, validation] = await Promise.all([
          fetchJson(DATA_PATHS.signal),
          fetchText(DATA_PATHS.performance),
          fetchText(DATA_PATHS.equity),
          fetchText(DATA_PATHS.weekly),
          fetchText(DATA_PATHS.trades),
          fetchText(DATA_PATHS.news),
          fetchJson(DATA_PATHS.validation).catch(() => null),
        ]);

        if (!cancelled) {
          setData({
            signal,
            performance: parseCsv(performanceText),
            equity: parseCsv(equityText),
            weekly: parseCsv(weeklyText),
            trades: parseCsv(tradesText),
            news: parseCsv(newsText),
            validation,
          });
        }
      } catch (loadError) {
        if (!cancelled) setError(loadError.message);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const derived = useMemo(() => {
    if (!data) return null;
    const strategy = data.performance.find((row) => row.name === 'strategy') ?? {};
    const benchmark = data.performance.find((row) => row.name === 'benchmark_hs300') ?? {};
    const defensive = data.performance.find((row) => row.name === 'defensive_static') ?? {};
    const lastEquity = latest(data.equity) ?? {};
    const lastWeekly = latest(data.weekly) ?? {};
    const closedTrades = data.trades.filter((trade) => trade.status === 'closed');
    const recentTrades = [...data.trades].slice(-8).reverse();
    const winners = closedTrades.filter((trade) => Number(trade.estimated_net_return) > 0).length;
    const avgTrade = closedTrades.reduce((sum, trade) => sum + Number(trade.estimated_net_return || 0), 0) / Math.max(closedTrades.length, 1);
    const holdingCounts = {};
    data.weekly.forEach((row) => {
      Object.entries(row).forEach(([key, value]) => {
        if (!key.startsWith('weight_') || Number(value) <= 0) return;
        const code = key.replace('weight_', '');
        holdingCounts[`${ETF_NAMES[code] ?? code} ${code}`] = (holdingCounts[`${ETF_NAMES[code] ?? code} ${code}`] ?? 0) + 1;
      });
    });
    const weeklyScoreRows = data.weekly.map((row) => ({
      date: row.date,
      market: Number(row.market_score || 0),
      technical: Number(row.technical_score || 0),
      news: Number(row.news_score || 50),
      policy: Number(row.policy_score || 50),
    }));

    return {
      strategy,
      benchmark,
      defensive,
      lastEquity,
      lastWeekly,
      recentTrades,
      closedTrades,
      winRate: winners / Math.max(closedTrades.length, 1),
      avgTrade,
      holdingCounts,
      weeklyScoreRows,
      drawdown: calcDrawdown(data.equity),
    };
  }, [data]);

  const drawdownOption = useMemo(() => {
    if (!derived) return {};
    return {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis', valueFormatter: (value) => `${Number(value).toFixed(2)}%` },
      grid: { left: 48, right: 18, top: 24, bottom: 32 },
      xAxis: {
        type: 'category',
        data: derived.drawdown.map((row) => row.time),
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8' },
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: '#94a3b8', formatter: '{value}%' },
        splitLine: { lineStyle: { color: 'rgba(148, 163, 184, 0.12)' } },
      },
      series: [
        {
          name: '策略回撤',
          type: 'line',
          smooth: true,
          showSymbol: false,
          areaStyle: { color: 'rgba(239, 68, 68, 0.18)' },
          lineStyle: { color: '#ef4444', width: 2 },
          data: derived.drawdown.map((row) => row.value),
        },
      ],
    };
  }, [derived]);

  const holdingOption = useMemo(() => {
    if (!derived) return {};
    return {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'item' },
      legend: {
        bottom: 0,
        textStyle: { color: '#aeb7c4' },
        type: 'scroll',
      },
      series: [
        {
          name: '持仓周数',
          type: 'pie',
          radius: ['46%', '72%'],
          center: ['50%', '42%'],
          itemStyle: { borderColor: '#0f172a', borderWidth: 2 },
          label: { color: '#dbe4f0', formatter: '{b}\n{d}%' },
          data: Object.entries(derived.holdingCounts).map(([name, value]) => ({ name, value })),
        },
      ],
    };
  }, [derived]);

  const scoreOption = useMemo(() => {
    if (!derived) return {};
    const rows = derived.weeklyScoreRows.slice(-80);
    return {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      grid: { left: 42, right: 16, top: 26, bottom: 32 },
      xAxis: {
        type: 'category',
        data: rows.map((row) => row.date),
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8' },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        axisLabel: { color: '#94a3b8' },
        splitLine: { lineStyle: { color: 'rgba(148, 163, 184, 0.12)' } },
      },
      series: [
        { name: '市场', type: 'line', smooth: true, showSymbol: false, data: rows.map((row) => row.market), color: '#22c55e' },
        { name: '技术', type: 'line', smooth: true, showSymbol: false, data: rows.map((row) => row.technical), color: '#60a5fa' },
        { name: '新闻', type: 'line', smooth: true, showSymbol: false, data: rows.map((row) => row.news), color: '#f59e0b' },
        { name: '政策', type: 'line', smooth: true, showSymbol: false, data: rows.map((row) => row.policy), color: '#a78bfa' },
      ],
    };
  }, [derived]);

  if (!data || !derived) return <LoadingPanel error={error} />;

  const { signal } = data;
  const target = signal.target ?? {};
  const scores = signal.scores ?? {};
  const validation = data.validation ?? {};
  const effectivePeriod = validation.effective_period ?? {};
  const split = validation.forward_split ?? {};

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">A 股 ETF 单仓轮动</span>
          <h1>策略执行与回测面板</h1>
          <p>主图采用专业行情图表，数据来自本地策略输出；热点只展示候选，不默认污染长期回测。</p>
        </div>
        <div className="signal-card">
          <span>最新信号 {signal.date}</span>
          <strong>{target.name ?? '--'} {target.code ?? ''}</strong>
          <div>
            <StatusPill bucket={signal.bucket} />
            <b>{fmtPct(target.weight ?? 0, 0)}</b>
          </div>
        </div>
      </header>

      <section className="metrics-grid">
        <MetricCard label="策略年化" value={fmtPct(derived.strategy.annual_return)} tone={pctClass(derived.strategy.annual_return)} hint={`总收益 ${fmtPct(derived.strategy.total_return)}`} />
        <MetricCard label="最大回撤" value={fmtPct(derived.strategy.max_drawdown)} tone="negative" hint={`Calmar ${fmtNum(derived.strategy.calmar)}`} />
        <MetricCard label="Sharpe" value={fmtNum(derived.strategy.sharpe)} hint={`胜率 ${fmtPct(derived.strategy.win_rate)}`} />
        <MetricCard label="年化换手" value={fmtNum(derived.strategy.annual_turnover, 1)} hint={`交易 ${derived.closedTrades.length} 次`} />
        <MetricCard label="沪深300年化" value={fmtPct(derived.benchmark.annual_return)} tone={pctClass(derived.benchmark.annual_return)} hint={`回撤 ${fmtPct(derived.benchmark.max_drawdown)}`} />
        <MetricCard label="防守基准年化" value={fmtPct(derived.defensive.annual_return)} tone={pctClass(derived.defensive.annual_return)} hint={`回撤 ${fmtPct(derived.defensive.max_drawdown)}`} />
      </section>

      <section className="panel hero-panel">
        <div className="panel-title">
          <div>
            <h2>净值走势</h2>
            <p>策略、沪深300、防守基准同轴对比</p>
          </div>
          <div className="quote-strip">
            <span>最新净值 <b>{fmtNum(derived.lastEquity.strategy, 3)}</b></span>
            <span>周换手 <b>{fmtPct(derived.lastEquity.turnover, 0)}</b></span>
            <span>手续费 <b>{fmtPct(derived.lastEquity.cost, 2)}</b></span>
          </div>
        </div>
        <TradingChart equityRows={data.equity} />
      </section>

      <section className="dashboard-grid">
        <article className="panel">
          <div className="panel-title">
            <h2>策略回撤</h2>
            <p>按策略净值滚动高点计算</p>
          </div>
          <EChart className="chart chart-md" option={drawdownOption} />
        </article>
        <article className="panel">
          <div className="panel-title">
            <h2>持仓分布</h2>
            <p>历史周频单仓占比</p>
          </div>
          <EChart className="chart chart-md" option={holdingOption} />
        </article>
      </section>

      <section className="dashboard-grid">
        <article className="panel">
          <div className="panel-title">
            <h2>因子分数</h2>
            <p>最近 80 周市场、技术、新闻、政策</p>
          </div>
          <EChart className="chart chart-md" option={scoreOption} />
        </article>
        <article className="panel">
          <div className="panel-title">
            <h2>当前因子</h2>
            <p>用于本次最新信号</p>
          </div>
          <div className="score-list">
            {Object.entries(scores).map(([key, value]) => (
              <div key={key}>
                <span>{key}</span>
                <strong>{fmtNum(value, 1)}</strong>
                <i style={{ width: `${Math.max(0, Math.min(100, Number(value)))}%` }} />
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="panel">
        <div className="panel-title">
          <div>
            <h2>最近交易</h2>
            <p>卖出时点、切换原因和估算扣费后收益</p>
          </div>
          <div className="quote-strip">
            <span>交易胜率 <b>{fmtPct(derived.winRate)}</b></span>
            <span>单笔均值 <b className={pctClass(derived.avgTrade)}>{fmtPct(derived.avgTrade)}</b></span>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>状态</th>
                <th>持仓</th>
                <th>买入</th>
                <th>卖出</th>
                <th>周期</th>
                <th>收益</th>
                <th>卖出原因</th>
                <th>下个标的</th>
              </tr>
            </thead>
            <tbody>
              {derived.recentTrades.map((trade) => (
                <tr key={`${trade.code}-${trade.buy_date}-${trade.sell_date}`}>
                  <td>{trade.status}</td>
                  <td>{trade.name} {trade.code}</td>
                  <td>{trade.buy_date}</td>
                  <td>{trade.sell_date || '--'}</td>
                  <td>{trade.holding_days || '--'} 天</td>
                  <td className={pctClass(trade.estimated_net_return)}>{fmtPct(trade.estimated_net_return)}</td>
                  <td>{trade.sell_reason || '--'}</td>
                  <td>{trade.next_name || '--'} {trade.next_code || ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="dashboard-grid">
        <article className="panel">
          <div className="panel-title">
            <h2>严谨性检查</h2>
            <p>先看样本证明力，再看收益</p>
          </div>
          <div className="validation-grid">
            <div>
              <span>实际样本</span>
              <strong>{effectivePeriod.start ?? '--'} 至 {effectivePeriod.end ?? '--'}</strong>
              <em>{effectivePeriod.weeks ?? '--'} 周 / {fmtNum(effectivePeriod.years, 2)} 年</em>
            </div>
            <div>
              <span>执行假设</span>
              <strong>{validation.method === 'rule_based_no_training' ? '规则策略，无训练模型' : '--'}</strong>
              <em>{validation.cost_bps ?? '--'} bps 手续费，下一期收益生效</em>
            </div>
            <div>
              <span>热点因子</span>
              <strong>{validation.uses_sentiment_in_backtest ? '已进入回测' : '未进入默认回测'}</strong>
              <em>{validation.sentiment_file || '热点只展示候选'}</em>
            </div>
          </div>
          <div className="warnings-list">
            {(validation.warnings ?? []).map((warning) => (
              <p key={warning}>{warning}</p>
            ))}
          </div>
        </article>
        <article className="panel">
          <div className="panel-title">
            <h2>本周路由</h2>
            <p>信号选择和候选防守标的</p>
          </div>
          <div className="route-grid">
            <div>
              <span>轮动选择</span>
              <strong>{signal.choices?.rotation?.name ?? '--'}</strong>
              <em>{signal.choices?.rotation?.code ?? ''}</em>
            </div>
            <div>
              <span>第一防守</span>
              <strong>{signal.choices?.defensive_first?.name ?? '--'}</strong>
              <em>{signal.choices?.defensive_first?.code ?? ''}</em>
            </div>
            <div>
              <span>回测档位</span>
              <strong>{derived.lastWeekly.bucket ?? '--'}</strong>
              <em>market {fmtNum(derived.lastWeekly.market_score)}</em>
            </div>
          </div>
        </article>
      </section>

      {split.enabled ? (
        <section className="dashboard-grid">
          <article className="panel">
            <div className="panel-title">
              <h2>前后段验证</h2>
              <p>固定参数，不在后段重新调参</p>
            </div>
            <div className="validation-grid two">
              <div>
                <span>前段 {split.early?.start} 至 {split.early?.end}</span>
                <strong>{fmtPct(split.early?.annual_return)}</strong>
                <em>回撤 {fmtPct(split.early?.max_drawdown)} / Sharpe {fmtNum(split.early?.sharpe)}</em>
              </div>
              <div>
                <span>后段 {split.late?.start} 至 {split.late?.end}</span>
                <strong>{fmtPct(split.late?.annual_return)}</strong>
                <em>回撤 {fmtPct(split.late?.max_drawdown)} / Sharpe {fmtNum(split.late?.sharpe)}</em>
              </div>
            </div>
          </article>
        <article className="panel">
          <div className="panel-title">
            <h2>热点候选</h2>
            <p>AkShare 拉取后仅作观察和审核</p>
          </div>
          <div className="news-list">
            {data.news.slice(0, 6).map((item, index) => (
              <div key={`${item.date}-${item.theme}-${index}`}>
                <span>{item.date}</span>
                <strong>{item.theme || item.title}</strong>
                <p>{item.note || item.title || '--'}</p>
              </div>
            ))}
          </div>
        </article>
      </section>
      ) : null}
    </main>
  );
}

export default App;
