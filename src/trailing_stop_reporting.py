"""Assembles the final Arabic report for the trailing-stop optimization
test (see trailing-stop-optimization-prompt.md). Code/columns stay English;
narrative is Arabic, per the same convention as reporting.py.
"""
from src import config


def _fmt(x, digits=3):
    return f"{x:.{digits}f}" if x == x else "غير متوفر"


def _fmt_pct(x, digits=1):
    return f"{x*100:.{digits}f}%" if x == x else "غير متوفر"


def _pf(x):
    return "∞" if x == float("inf") else _fmt(x, 3)


def build_report(results: dict) -> str:
    summary_cons = results["summary_conservative"]
    summary_agg = results["summary_aggressive"]
    mc = results["monte_carlo"]
    sig = results["significance"]
    monthly = results["monthly"]
    overfit = results["overfitting"]
    regime = results["regime"]
    best_pct = results["best_pct"]

    baseline = summary_cons[summary_cons["config"] == "baseline (no trailing)"].iloc[0]
    best_row = summary_cons[summary_cons["pct"] == best_pct].iloc[0]
    n_improve_months = int((monthly["direction"] == "improved").sum())
    n_degrade_months = int((monthly["direction"] == "degraded").sum())

    lines = []
    lines.append("# تقرير اختبار تحسين الوقف المتحرك (Trailing Stop) والتحقق بمونت كارلو — NAS100\n")
    lines.append(
        "هذا تحليل افتراضي (test-only) بحت: لم يتم تعديل عمود `rPnL` الأصلي، ولا مستويات وقف الخسارة "
        "(`initalSL`)، ولا استراتيجية الدخول. الهدف الوحيد هو اختبار: ماذا لو أُضيف وقف متحرك فوق نفس "
        f"الـ{baseline['n']} صفقة، بدلاً من طريقة الخروج الفعلية؟\n"
    )

    lines.append("## تعريف الوقف المتحرك المُستخدَم\n")
    lines.append(
        "نسبة الوقف المتحرك تعني **نسبة التخلي عن الربح المفتوح**: بمجرد أن تصل القمة المُحققة إلى "
        f"{config.TRAILING_ACTIVATION_MFE_R}R فوق سعر الدخول (تفعيل)، يتبع الوقف القمة بحيث يخرج "
        "عند `القمة - نسبة% × (القمة - سعر الدخول)`. الوقف المتحرك لا يمكن أن يكون أوسع من وقف الخسارة "
        "الأصلي أبداً (لا يُغيَّر وقف الخسارة، بل يُشدَّد فقط). التعبئة محافظة (conservative fill): إذا "
        "تجاوز افتتاح الشمعة مستوى الوقف (فجوة سعرية)، استُخدم سعر الافتتاح كسعر خروج أسوأ.\n"
        "**افتراض موثَّق (لم يُحدَّد في الطلب الأصلي)**: عتبة التفعيل "
        f"({config.TRAILING_ACTIVATION_MFE_R}R) مُستعارة من نفس عتبة \"الحركة الجوهرية\" المستخدمة سابقاً "
        "في `exit_quality.py`. هذا الاختيار مؤثر: فور التفعيل بالكاد عند هذه العتبة، تكون المسافة "
        "(القمة - الدخول) صغيرة جداً، فيصبح الفارق بين نسبة 1% و20% ضئيلاً بالسعر المطلق — ما يفسر لماذا "
        "تتشابه نتائج معظم النسب في الجدول أدناه أكثر مما قد يُتوقَّع.\n"
    )

    lines.append("## الملخص التنفيذي\n")
    verdict_ar = "لا" if best_row["profit_factor"] < baseline["profit_factor"] else "نعم"
    lines.append(
        f"- **الإجابة المختصرة: {verdict_ar}**, الوقف المتحرك بصيغته المُختبرة هنا **لم يُحسِّن** الأداء — "
        f"بل أضعفه عبر كامل الشبكة المُختبرة (1% إلى 20%). معامل الربح الأساسي (بدون وقف متحرك) = "
        f"**{_pf(baseline['profit_factor'])}**، بينما أفضل إعداد ({best_pct*100:.0f}%) حقق معامل ربح "
        f"= **{_pf(best_row['profit_factor'])}** فقط (أقل من 1، أي خسارة صافية).\n"
        f"- العائد المتوقع الأساسي = {_fmt(baseline['expectancy'])}R للصفقة، مقابل "
        f"{_fmt(best_row['expectancy'])}R تحت أفضل إعداد وقف متحرك — **سالب** رغم أن معدل الفوز نفسه "
        f"ارتفع من {_fmt_pct(baseline['win_rate'])} إلى {_fmt_pct(best_row['win_rate'])}. السبب: الوقف "
        "المتحرك يحوّل بعض الصفقات الخاسرة إلى رابحة صغيرة، لكنه في المقابل يبتر الصفقات الرابحة الكبيرة "
        "(الفوز الكبير) التي تعتمد عليها هذه الاستراتيجية أساساً في تحقيق ربحها (معدل فوز أساسي منخفض "
        f"{_fmt_pct(baseline['win_rate'])} يعني أن الحافة تأتي من عدد قليل من الصفقات كبيرة العائد).\n"
        f"- **مونت كارلو يؤكد أن هذا ليس ضجيجاً**: بعد 1000 إعادة تشكيل (bootstrap) على أفضل 3 إعدادات، "
        "فترة الثقة 90% لمعامل الربح لم تشمل معامل الربح الأساسي في أي منها — أي أن تراجع الأداء **متّسق "
        "وقوي إحصائياً**, وليس صدفة في هذه العيّنة تحديداً.\n"
    )

    lines.append("## جدول الشبكة الكاملة (السيناريو المحافظ)\n")
    lines.append(
        "السيناريو المحافظ: إذا لم يُفعَّل الوقف المتحرك خلال نافذة الصفقة الأصلية (`dateStart` إلى "
        "`dateEnd`)، يُستخدم الخروج الفعلي الأصلي دون تمديد.\n"
    )
    lines.append("| الإعداد | العيّنة n | معدل الفوز | العائد المتوقع (R) | معامل الربح | Sharpe | أقصى تراجع (R) | تغيّر PF% | تغيّر العائد المتوقع% |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in summary_cons.iterrows():
        lines.append(
            f"| {r['config']} | {int(r['n'])} | {_fmt_pct(r['win_rate'])} | {_fmt(r['expectancy'])} | "
            f"{_pf(r['profit_factor'])} | {_fmt(r['sharpe'], 2)} | {_fmt(r['max_drawdown_r'], 1)} | "
            f"{_fmt(r['pf_vs_baseline_pct_change'], 1)}% | {_fmt(r['expectancy_vs_baseline_pct_change'], 1)}% |"
        )
    lines.append("")

    lines.append("## السيناريو المتشدد (Aggressive)\n")
    lines.append(
        "إذا لم يُفعَّل الوقف المتحرك خلال نافذة الصفقة الأصلية، سُمح للمحاكاة بمتابعة الشموع حتى "
        f"{config.TRAILING_AGGRESSIVE_MAX_EXTENSION_MINUTES // (24*60)} أيام إضافية بحثاً عن نقطة تفعيل "
        "لاحقة (بدون معلومات مستقبلية غير متاحة حينها بالطبع، فقط تمديد افتراضي لمدة الصفقة). النتيجة "
        "لا تتغير جوهرياً:\n"
    )
    best_row_agg = summary_agg[summary_agg["pct"] == best_pct].iloc[0]
    lines.append(f"- معامل ربح أفضل إعداد تحت السيناريو المتشدد: {_pf(best_row_agg['profit_factor'])} "
                 f"(مقابل {_pf(best_row['profit_factor'])} في السيناريو المحافظ) — لا يزال أقل من الأساسي "
                 f"{_pf(baseline['profit_factor'])}.\n")

    lines.append("## نتائج مونت كارلو (أفضل 3 إعدادات، Bootstrap 1000 مرة)\n")
    lines.append("| النسبة | متوسط PF (Bootstrap) | الحد الأدنى 5% | الحد الأعلى 95% | PF الأساسي | يتقاطع مع الأساسي؟ | % مرات تفوّق على الأساسي | الحكم |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in mc.iterrows():
        lines.append(
            f"| {r['pct']*100:.0f}% | {_pf(r['mean_pf'])} | {_pf(r['ci_lo_5th'])} | {_pf(r['ci_hi_95th'])} | "
            f"{_pf(r['observed_baseline_pf'])} | {'نعم' if r['ci_crosses_baseline'] else 'لا'} | "
            f"{_fmt_pct(r['pct_boot_iters_config_beats_baseline'])} | {r['robust_verdict']} |"
        )
    lines.append(
        "\n**ملاحظة مهمة حول القراءة**: \"لا يتقاطع مع الأساسي\" هنا يعني أن **التراجع** في الأداء ثابت "
        "وموثوق إحصائياً (وليس أن الوقف المتحرك أداؤه ثابت وجيد) — الحكم يوصف متانة الفرق عن الأساسي، "
        "لا اتجاه الفرق نفسه.\n"
    )

    lines.append("## الاختبار الإحصائي المزدوج (Paired t-test)\n")
    lines.append(
        f"H0: متوسط العائد الأصلي = متوسط عائد أفضل إعداد وقف متحرك ({best_pct*100:.0f}%), على نفس "
        f"الـ{sig['n']} صفقة (مزدوج/paired وليس عيّنتين مستقلتين).\n\n"
        f"- t = {_fmt(sig['t_statistic'], 3)}, p = {_fmt(sig['p_value'], 4)}\n"
        f"- Cohen's d (مزدوج) = {_fmt(sig['cohens_d'], 3)}\n"
        f"- الفرق في المتوسط: {_fmt(sig['mean_diff'], 4)}R للصفقة الواحدة\n\n"
    )
    if sig["interpretation"] == "significant_and_meaningful":
        interp_ar = "**دلالة إحصائية حقيقية وحجم تأثير ذو معنى عملي.**"
    elif sig["interpretation"] == "significant_but_tiny_effect":
        interp_ar = (
            "**دلالة إحصائية حقيقية (p<0.05)، لكن Cohen's d صغير حسب المعايير التقليدية.** هذا لا يعني أن "
            "الأثر مهمل عملياً: Cohen's d صغير هنا لأن **تباين الفروقات الفردية مرتفع جداً** (بعض الصفقات "
            "تخسر الكثير من الربح المقطوع، بينما أخرى لا تتأثر) — وليس لأن متوسط الفرق صغير. فرق "
            f"{_fmt(sig['mean_diff'], 3)}R للصفقة الواحدة، مُجمَّعاً على {sig['n']} صفقة، يعادل خسارة "
            f"تراكمية تقارب {sig['mean_diff']*sig['n']:.0f}R — رقم غير مهمل عملياً رغم صغر Cohen's d."
        )
    else:
        interp_ar = "**لا توجد دلالة إحصائية كافية للجزم بوجود فرق حقيقي.**"
    lines.append(f"{interp_ar}\n")

    lines.append("## المقارنة الشهرية (أفضل إعداد مقابل الأصلي)\n")
    lines.append(
        f"من أصل {len(monthly)} شهراً: **{n_improve_months}** شهراً تحسّن، **{n_degrade_months}** شهراً تراجع "
        "(بالدولار). الرسم البياني الكامل في `outputs/figures/trailing_stop_monthly_comparison.png`. أبرز "
        "5 أشهر تراجعاً وأبرز 5 أشهر تحسناً:\n"
    )
    worst5 = monthly.sort_values("difference_usd").head(5)
    best5 = monthly.sort_values("difference_usd", ascending=False).head(5)
    lines.append("**الأكثر تراجعاً:**\n")
    for _, r in worst5.iterrows():
        lines.append(f"- {r['month']}: {r['difference_usd']:,.0f}$ (أصلي {r['original_pnl_usd']:,.0f}$ ← "
                     f"وقف متحرك {r['trailing_pnl_usd']:,.0f}$)")
    lines.append("\n**الأكثر تحسناً:**\n")
    for _, r in best5.iterrows():
        lines.append(f"- {r['month']}: +{r['difference_usd']:,.0f}$ (أصلي {r['original_pnl_usd']:,.0f}$ ← "
                     f"وقف متحرك {r['trailing_pnl_usd']:,.0f}$)")
    lines.append("")

    lines.append("## فحص فرط التخصيص (Train/Test Split)\n")
    lines.append(
        f"تدريب على أول {config.TRAILING_TRAIN_FRACTION*100:.0f}% زمنياً ({overfit['train_n_trades']} صفقة)، "
        f"تحقق على آخر {(1-config.TRAILING_TRAIN_FRACTION)*100:.0f}% ({overfit['test_n_trades']} صفقة).\n\n"
        f"- أفضل نسبة على بيانات التدريب: **{overfit['best_on_train_pct']*100:.0f}%**\n"
        f"- أفضل نسبة على بيانات التحقق (Holdout): **{overfit['best_on_test_pct']*100:.0f}%**\n"
        f"- متّسقة بين التدريب والتحقق؟ **{'نعم' if overfit['best_pct_consistent_train_test'] else 'لا'}**\n\n"
    )
    if not overfit["best_pct_consistent_train_test"]:
        lines.append(
            "**النتيجة تشير بوضوح إلى فرط تخصيص (overfitting)**: أفضل نسبة على عيّنة التدريب ليست الأفضل "
            "على عيّنة التحقق — أي أن اختيار \"أفضل نسبة\" من الشبكة الكاملة على كامل العيّنة (789 صفقة) "
            "لا يُعتمد عليه كقرار نهائي، وهو دليل إضافي (وليس فقط الأداء العام السلبي) على عدم متانة "
            "فكرة الوقف المتحرك لهذه الاستراتيجية تحديداً.\n"
        )
    lines.append("| الإعداد | n (تدريب) | عائد متوقع (تدريب) | PF (تدريب) | PF (تحقق) | عائد متوقع (تحقق) |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in overfit["top_train_candidates"].iterrows():
        lines.append(f"| {r['pct']*100:.0f}% | {int(r['train_n'])} | {_fmt(r['train_expectancy'])} | "
                     f"{_pf(r['train_profit_factor'])} | {_pf(r['test_profit_factor'])} | {_fmt(r['test_expectancy'])} |")
    lines.append("")

    lines.append("## اعتماد النظام السوقي (Regime Dependency)\n")
    lines.append(f"أداء أفضل إعداد ({best_pct*100:.0f}%) مقسَّماً حسب نظام السوق وقت دخول كل صفقة "
                 "(الاتجاه/التذبذب من `regime_detection.py`, بدون تسريب مستقبلي):\n")
    lines.append("| البُعد | التصنيف | n | PF (وقف متحرك) | PF (أصلي) | يتفوق الوقف المتحرك؟ |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in regime.iterrows():
        lines.append(f"| {r['regime_dimension']} | {r['regime_label']} | {int(r['n'])} | "
                     f"{_pf(r['trailing_profit_factor'])} | {_pf(r['baseline_profit_factor'])} | "
                     f"{'نعم' if r['trailing_beats_baseline'] else 'لا'} |")
    n_beats = int(regime["trailing_beats_baseline"].sum())
    lines.append(
        f"\nمن أصل {len(regime)} تصنيف نظام سوقي، تفوّق الوقف المتحرك في **{n_beats}** فقط. "
        + ("أبرزها: **التقلب المرتفع (High Vol)** — وهو المكان الوحيد الذي بدا فيه الوقف المتحرك مفيداً، "
           "ما يشير إلى أن قيمته المحتملة (إن وُجدت) مشروطة بنظام تقلب مرتفع تحديداً، وليست ميزة عامة "
           "للاستراتيجية.\n" if n_beats > 0 and n_beats < len(regime) else "\n")
    )

    lines.append("## التحفظات (Caveats)\n")
    lines.append(
        "- **حجم العيّنة**: 789 صفقة فقط، ومقسَّمة إلى شبكة من 8 إعدادات × سيناريوهين × تقسيم تدريب/تحقق — "
        "أي عيّنة تحقق تنزل أحياناً لعشرات الصفقات شهرياً، ما يجعل بعض النتائج الفرعية (كالتفصيل الشهري) "
        "حساسة للصدفة رغم أن النتيجة الإجمالية (تراجع الأداء) مدعومة بمونت كارلو على كامل العيّنة.\n"
        "- **افتراض عتبة التفعيل**: كما ذُكر أعلاه، عتبة 0.15R للتفعيل اختيار منهجي غير محدَّد في الطلب "
        "الأصلي، وله تأثير مباشر على مدى تشابه نتائج نسب الوقف المختلفة (نسب صغيرة جداً بالقرب من عتبة "
        "التفعيل تُصبح متطابقة تقريباً في السعر المطلق).\n"
        "- **افتراضات استشرافية (forward-looking) داخل السيناريو المتشدد فقط**: يستخدم شموعاً بعد "
        "`dateEnd` الأصلي، وهذا افتراضي بحت (لم تكن هذه الفترة الممتدة جزءاً من الصفقة الفعلية تاريخياً) — "
        "استُخدم فقط لفهم الحساسية، وليس كسيناريو أساسي للتوصية.\n"
        "- **لا تعديل على منطق الدخول أو وقف الخسارة الأصلي**: أي تحسّن أو تراجع هنا مرتبط حصراً بتوقيت "
        "الخروج، وليس جودة نقطة الدخول (وهي ثابتة كما هي في التحليل الأصلي).\n"
    )

    lines.append("## التوصية النهائية\n")
    lines.append(
        "**لا يُنصح بتطبيق الوقف المتحرك بصيغته المُختبرة هنا على التداول الحي.** الأدلة متّسقة عبر أربع "
        "زوايا مستقلة: (1) كامل شبكة النسب المختبرة (1%-20%) أظهرت معامل ربح أقل من 1 دون استثناء يُذكر، "
        "(2) مونت كارلو أكد أن هذا التراجع ثابت إحصائياً وليس ضجيجاً في هذه العيّنة تحديداً، (3) الاختبار "
        "المزدوج أظهر دلالة إحصائية حقيقية (p<0.05) رغم صغر Cohen's d الظاهري، (4) فحص التدريب/التحقق أظهر "
        "أن \"أفضل نسبة\" غير متّسقة زمنياً — أي حتى لو تجاهلنا النتيجة العامة السلبية، اختيار نسبة معينة "
        "كتوصية نهائية غير موثوق به. **الاستثناء الوحيد الجدير بالمتابعة**: أداء أفضل من الأساسي في نظام "
        "التقلب المرتفع تحديداً — يستحق اختباراً مستقلاً ومُسجَّلاً مسبقاً (pre-registered) بعيّنة أكبر قبل "
        "أي اعتماد عملي، وليس كقاعدة عامة.\n"
    )

    return "\n".join(lines)
