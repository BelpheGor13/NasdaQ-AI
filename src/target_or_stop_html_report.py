"""Generates the self-contained, interactive HTML version of the
target-or-stop report (KPI cards, filterable/sortable per-trade table,
monthly summary) -- the "official" format matching the earlier session's
report, rebuilt from the current codebase with the same-candle tie-
breaking caveat now quantified instead of left implicit.

No external assets/CDNs: everything (CSS, JS, data) is inlined so the file
opens standalone in a browser, offline.
"""
import json

from src import config, data_loading, exit_strategy_simulation, target_or_stop_data

REPORT_PATH = config.REPORTS_OUT / "target_or_stop_interactive_report.html"

STATUS_LABELS_AR = {
    "missed_profit": "فوّتّ ربحًا إضافيًا",
    "protected": "التأمين حماك",
    "flipped_to_loss": "انقلبت من ربح لخسارة كاملة",
    "no_diff": "بدون فرق",
}
SIDE_LABELS_AR = {"buy": "شراء", "sell": "بيع"}


def _fmt_usd(x: float) -> str:
    sign = "+" if x >= 0 else "-"
    return f"{sign}${abs(x):,.0f}"


def build_html(trade_table, monthly_table, kpis: dict, tie_worst_case_usd: float) -> str:
    trades_json = json.dumps([
        {
            "date": r.date, "side": SIDE_LABELS_AR[r.side],
            "actual_r": round(r.actual_r, 2), "target_r": round(r.target_r, 2),
            "actual_usd": round(r.actual_usd, 0), "target_usd": round(r.target_usd, 0),
            "diff_usd": round(r.diff_usd, 0), "status": r.status,
            "status_ar": STATUS_LABELS_AR[r.status], "tie": bool(r.same_candle_tie),
        }
        for r in trade_table.itertuples()
    ], ensure_ascii=False)

    monthly_json = json.dumps([
        {"month": r.month, "n": int(r.n), "actual_usd": round(r.actual_usd, 0),
         "target_usd": round(r.target_usd, 0), "diff_usd": round(r.diff_usd, 0)}
        for r in monthly_table.itertuples()
    ], ensure_ascii=False)

    total_diff = kpis["total_diff_usd"]
    worst_case_total = total_diff + tie_worst_case_usd

    html = f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>هل التأمين حماك أم كلّفك؟</title>
<style>
  :root {{
    --green: #1e8e5a; --red: #d64545; --gray: #6b7280; --bg-card: #f7f8fa;
    --border: #e3e5e8; --accent: #2f5fb3;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Segoe UI", Tahoma, Arial, sans-serif; margin: 0; padding: 24px 32px 60px;
    background: #ffffff; color: #1c1f24; line-height: 1.6;
  }}
  h1 {{ font-size: 1.6rem; margin-bottom: 6px; }}
  p.subtitle {{ color: var(--gray); max-width: 900px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 22px 0; }}
  .kpi-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; text-align: center; }}
  .kpi-card .label {{ font-size: 0.82rem; color: var(--gray); margin-bottom: 8px; }}
  .kpi-card .value {{ font-size: 1.5rem; font-weight: 700; }}
  .kpi-card.pos .value {{ color: var(--green); }}
  .kpi-card.neg .value {{ color: var(--red); }}
  .caveat {{
    background: #fff7e6; border: 1px solid #f0d58c; border-radius: 10px; padding: 14px 18px;
    margin: 20px 0; font-size: 0.92rem;
  }}
  .caveat b {{ color: #92650a; }}
  h2 {{ font-size: 1.15rem; margin-top: 34px; border-bottom: 2px solid var(--border); padding-bottom: 6px; }}
  .chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 14px 0; }}
  .chip {{
    border: 1px solid var(--border); background: #fff; border-radius: 999px; padding: 6px 14px;
    cursor: pointer; font-size: 0.85rem; user-select: none;
  }}
  .chip.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 8px; }}
  th, td {{ padding: 8px 10px; text-align: center; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  th {{ cursor: pointer; background: var(--bg-card); position: sticky; top: 0; font-weight: 600; }}
  th:hover {{ background: #edf0f4; }}
  tr:hover td {{ background: #fafbfc; }}
  .pos {{ color: var(--green); }}
  .neg {{ color: var(--red); }}
  .badge {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.78rem; }}
  .badge.missed_profit {{ background: #e6f4ec; color: var(--green); }}
  .badge.protected {{ background: #eef1fb; color: var(--accent); }}
  .badge.flipped_to_loss {{ background: #fdeaea; color: var(--red); }}
  .badge.no_diff {{ background: #f0f1f3; color: var(--gray); }}
  .tie-flag {{ color: #92650a; font-size: 0.75rem; margin-right: 4px; }}
  .table-wrap {{ max-height: 520px; overflow: auto; border: 1px solid var(--border); border-radius: 8px; }}
  .table-wrap table {{ margin-top: 0; }}
  select {{ padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border); font-family: inherit; }}
  footer {{ margin-top: 40px; color: var(--gray); font-size: 0.8rem; border-top: 1px solid var(--border); padding-top: 14px; }}
</style>
</head>
<body>

<h1>هل التأمين (وقف الخسارة) حماك أم كلّفك؟</h1>
<p class="subtitle">
  مقارنة النتيجة الفعلية لكل صفقة (اللي صار فيها إغلاق/تأمين يدوي) بالنتيجة الافتراضية لو تُركت الصفقة
  تمشي لحد هدفها الأصلي (<code>idealTP</code>) أو وقف خسارتها الأصلي (<code>initalSL</code>) فقط، بدون أي
  تدخّل يدوي. {kpis['n_total']} صفقة، 2020–2024. الكود والمنهجية في
  <code>src/exit_strategy_simulation.py</code> + <code>src/target_or_stop_data.py</code>.
</p>

<div class="kpi-grid">
  <div class="kpi-card {'pos' if total_diff>=0 else 'neg'}"><div class="label">إجمالي الفرق لو التزمت بالهدف دائمًا</div><div class="value">{_fmt_usd(total_diff)}</div></div>
  <div class="kpi-card"><div class="label">أشهر تحسّنت</div><div class="value">{kpis['n_months_improved']} / {kpis['n_months']}</div></div>
  <div class="kpi-card pos"><div class="label">صفقات فوّتّ فيها ربح إضافي</div><div class="value">{kpis['n_missed_profit']}</div></div>
  <div class="kpi-card"><div class="label">صفقات التأمين حماك فيها</div><div class="value">{kpis['n_protected']}</div></div>
  <div class="kpi-card neg"><div class="label">منها: انقلبت من ربح لخسارة كاملة</div><div class="value">{kpis['n_flipped_to_loss']}</div></div>
</div>

<div class="caveat">
  <b>تحفّظ مُقاس رقميًا:</b> في {kpis['n_same_candle_ties']} صفقة فقط ({kpis['pct_same_candle_ties_of_resolved']*100:.2f}%
  من الصفقات المحسومة)، وقف الخسارة والهدف كانا داخل نفس شمعة الدقيقة الواحدة — البيانات لا تسمح بمعرفة
  أيّهما لُمس فعليًا أولًا، والمحاكاة ترجّح الهدف في هذه الحالة تحديدًا. الأثر الأسوأ الممكن لو كانت
  كل هذه الحالات التسع فعليًا وقف خسارة كامل بدل الهدف: الرقم الإجمالي يتحرك من
  <b>{_fmt_usd(total_diff)}</b> إلى <b>{_fmt_usd(worst_case_total)}</b>
  ({tie_worst_case_usd/total_diff*100:+.1f}%) — أثر هامشي لا يغيّر الخلاصة العامة.
  الصفوف المتأثرة مُعلَّمة بـ <span class="tie-flag">⚠ تعارض</span> في الجدول أدناه.
</div>

<h2>الملخص الشهري</h2>
<p class="subtitle">العائد الشهري بالدولار: الفعلي مقابل لو التزمت بالهدف. اضغط أي عمود للترتيب.</p>
<div class="table-wrap">
<table id="monthlyTable">
  <thead><tr>
    <th data-key="month" data-type="str">الشهر</th>
    <th data-key="n" data-type="num">عدد الصفقات</th>
    <th data-key="actual_usd" data-type="num">الفعلي ($)</th>
    <th data-key="target_usd" data-type="num">لو التزمت بالهدف ($)</th>
    <th data-key="diff_usd" data-type="num">الفرق ($)</th>
  </tr></thead>
  <tbody></tbody>
</table>
</div>

<h2>تفاصيل كل صفقة</h2>
<div class="chips" id="statusChips">
  <div class="chip active" data-filter="all">الكل ({kpis['n_total']})</div>
  <div class="chip" data-filter="missed_profit">فوّتّ ربح ({kpis['n_missed_profit']})</div>
  <div class="chip" data-filter="protected_all">التأمين حماك ({kpis['n_protected']})</div>
  <div class="chip" data-filter="flipped_to_loss">انقلبت لخسارة كاملة ({kpis['n_flipped_to_loss']})</div>
  <div class="chip" data-filter="no_diff">بدون فرق ({kpis['n_no_diff']})</div>
  <div class="chip" data-filter="tie">تعارض بنفس الشمعة ({kpis['n_same_candle_ties']})</div>
</div>
<select id="monthFilter"><option value="all">كل الأشهر</option></select>
<div class="table-wrap">
<table id="tradesTable">
  <thead><tr>
    <th data-key="date" data-type="str">التاريخ</th>
    <th data-key="side" data-type="str">الاتجاه</th>
    <th data-key="actual_r" data-type="num">R الفعلي</th>
    <th data-key="target_r" data-type="num">R لو التزمت</th>
    <th data-key="actual_usd" data-type="num">PnL الفعلي ($)</th>
    <th data-key="target_usd" data-type="num">PnL لو التزمت ($)</th>
    <th data-key="diff_usd" data-type="num">الفرق ($)</th>
    <th data-key="status" data-type="str">الحالة</th>
  </tr></thead>
  <tbody></tbody>
</table>
</div>

<footer>
  مبني على محاكاة استرجاعية للشموع الدقيقة (بدون تسريب معلومات مستقبلية) — سعر التنفيذ الافتراضي عند
  <code>idealTP</code> يفترض تنفيذًا مثاليًا بدون انزلاق سعري. <code>idealTP</code> هو أفضل سعر تحقّق
  فعليًا بأثر رجعي وليس هدفًا كان معروفًا مسبقًا قبل الدخول — راجع
  <code>outputs/reports/target_or_stop_report_arabic.md</code> للتحليل الإحصائي الكامل (اختبار مزدوج
  ومونت كارلو). الأرقام لأغراض البحث والتخطيط فقط، وليست توصية استثمارية.
</footer>

<script>
const trades = {trades_json};
const monthly = {monthly_json};

function fmtUsd(x) {{
  const sign = x >= 0 ? '+' : '-';
  return sign + '$' + Math.abs(Math.round(x)).toLocaleString('en-US');
}}
function cls(x) {{ return x >= 0 ? 'pos' : 'neg'; }}

let monthlySort = {{key: 'month', dir: 1}};
function renderMonthly() {{
  const rows = [...monthly].sort((a, b) => {{
    const k = monthlySort.key;
    return (a[k] > b[k] ? 1 : a[k] < b[k] ? -1 : 0) * monthlySort.dir;
  }});
  document.querySelector('#monthlyTable tbody').innerHTML = rows.map(r => `
    <tr>
      <td>${{r.month}}</td><td>${{r.n}}</td>
      <td class="${{cls(r.actual_usd)}}">${{fmtUsd(r.actual_usd)}}</td>
      <td class="${{cls(r.target_usd)}}">${{fmtUsd(r.target_usd)}}</td>
      <td class="${{cls(r.diff_usd)}}">${{fmtUsd(r.diff_usd)}}</td>
    </tr>`).join('');
}}
document.querySelectorAll('#monthlyTable th').forEach(th => th.addEventListener('click', () => {{
  const key = th.dataset.key;
  monthlySort.dir = (monthlySort.key === key) ? -monthlySort.dir : 1;
  monthlySort.key = key;
  renderMonthly();
}}));

const months = [...new Set(trades.map(t => t.date.slice(0, 7)))].sort();
const monthSelect = document.getElementById('monthFilter');
months.forEach(m => {{
  const opt = document.createElement('option');
  opt.value = m; opt.textContent = m;
  monthSelect.appendChild(opt);
}});

let activeFilter = 'all';
let activeMonth = 'all';
let tradesSort = {{key: 'date', dir: 1}};

function statusBadge(t) {{
  const tie = t.tie ? '<span class="tie-flag">⚠ تعارض</span>' : '';
  return tie + `<span class="badge ${{t.status}}">${{t.status_ar}}</span>`;
}}

function renderTrades() {{
  let rows = trades.filter(t => {{
    if (activeMonth !== 'all' && t.date.slice(0, 7) !== activeMonth) return false;
    if (activeFilter === 'all') return true;
    if (activeFilter === 'protected_all') return t.status === 'protected' || t.status === 'flipped_to_loss';
    if (activeFilter === 'tie') return t.tie;
    return t.status === activeFilter;
  }});
  rows = [...rows].sort((a, b) => {{
    const k = tradesSort.key;
    return (a[k] > b[k] ? 1 : a[k] < b[k] ? -1 : 0) * tradesSort.dir;
  }});
  document.querySelector('#tradesTable tbody').innerHTML = rows.map(t => `
    <tr>
      <td>${{t.date}}</td><td>${{t.side}}</td>
      <td class="${{cls(t.actual_r)}}">${{t.actual_r.toFixed(2)}}R</td>
      <td class="${{cls(t.target_r)}}">${{t.target_r.toFixed(2)}}R</td>
      <td class="${{cls(t.actual_usd)}}">${{fmtUsd(t.actual_usd)}}</td>
      <td class="${{cls(t.target_usd)}}">${{fmtUsd(t.target_usd)}}</td>
      <td class="${{cls(t.diff_usd)}}">${{fmtUsd(t.diff_usd)}}</td>
      <td>${{statusBadge(t)}}</td>
    </tr>`).join('');
}}
document.querySelectorAll('#tradesTable th').forEach(th => th.addEventListener('click', () => {{
  const key = th.dataset.key;
  tradesSort.dir = (tradesSort.key === key) ? -tradesSort.dir : 1;
  tradesSort.key = key;
  renderTrades();
}}));
document.querySelectorAll('#statusChips .chip').forEach(chip => chip.addEventListener('click', () => {{
  document.querySelectorAll('#statusChips .chip').forEach(c => c.classList.remove('active'));
  chip.classList.add('active');
  activeFilter = chip.dataset.filter;
  renderTrades();
}}));
monthSelect.addEventListener('change', () => {{ activeMonth = monthSelect.value; renderTrades(); }});

renderMonthly();
renderTrades();
</script>
</body>
</html>
"""
    return html


def main():
    config.REPORTS_OUT.mkdir(parents=True, exist_ok=True)

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = exit_strategy_simulation.simulate_exit_strategies(trades, candles)
    tie_ids = target_or_stop_data.find_same_candle_ties(trades, candles)

    trade_table = target_or_stop_data.build_trade_table(sim, tie_ids)
    monthly_table = target_or_stop_data.build_monthly_table(trade_table)
    kpis = target_or_stop_data.build_kpis(trade_table, monthly_table, tie_ids)

    tie_rows = trade_table[trade_table["id"].isin(tie_ids)]
    risk_usd = (trades.set_index("id").loc[tie_rows["id"], "entryPrice"] -
                trades.set_index("id").loc[tie_rows["id"], "initalSL"]).abs() * \
               trades.set_index("id").loc[tie_rows["id"], "amount"]
    worst_case_target_usd = -risk_usd.values
    tie_worst_case_usd = float((worst_case_target_usd - tie_rows["target_usd"].values).sum())

    html = build_html(trade_table, monthly_table, kpis, tie_worst_case_usd)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"Report: {REPORT_PATH}")
    for k, v in kpis.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
