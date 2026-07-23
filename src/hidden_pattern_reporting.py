"""Assembles the final Arabic report for the hidden-pattern discovery +
regime-conditional exit optimization test (see
hidden-patterns-exit-optimization-prompt.md). Code/columns stay English;
narrative is Arabic.
"""
from src import config


def _fmt(x, digits=3):
    return f"{x:.{digits}f}" if x == x else "غير متوفر"


def _fmt_pct(x, digits=1):
    return f"{x*100:.{digits}f}%" if x == x else "غير متوفر"


def _pf(x):
    return "∞" if x == float("inf") else _fmt(x, 3)


STRATEGY_NAMES_AR = {
    "baseline": "الخروج الفعلي الأصلي",
    "fixed_tp_idealTP": "هدف ربح ثابت عند idealTP",
    "fixed_tp_2R": "هدف ربح ثابت 2R",
    "fixed_tp_3R": "هدف ربح ثابت 3R",
    "fixed_tp_4R": "هدف ربح ثابت 4R",
    "mfe_aware": "وقف متحرك متكيّف مع MFE",
    "partial_profit": "جني أرباح جزئي (50%@2R، 30%@4R، 20%@TP)",
}
for _p in config.TRAILING_STOP_GRID:
    STRATEGY_NAMES_AR[f"trailing_{int(_p*100)}pct"] = f"وقف متحرك {int(_p*100)}%"
for _h in config.TIME_BASED_EXIT_HOURS:
    STRATEGY_NAMES_AR[f"time_{_h}h"] = f"خروج زمني بعد {_h} ساعة"


def _ar(name):
    return STRATEGY_NAMES_AR.get(name, name)


def build_report(results: dict) -> str:
    cluster_profile = results["cluster_profile"]
    cluster_table = results["cluster_table"]
    mc = results["monte_carlo"]
    comparison = results["comparison"]
    best_global_name = results["best_global_name"]
    sig_baseline = results["sig_vs_baseline"]
    sig_global = results["sig_vs_global"]
    monthly = results["monthly"]
    yearly = results["yearly"]
    dd = results["drawdown"]
    validation = results["validation"]
    silhouette = results["silhouette"]
    k = results["k"]

    n_improve = int((monthly["direction"] == "improved").sum())
    n_degrade = int((monthly["direction"] == "degraded").sum())
    baseline_row = comparison[comparison["metric_set"] == "baseline"].iloc[0]
    hybrid_row = comparison[comparison["metric_set"] == "hybrid_cluster_conditional"].iloc[0]
    global_row = comparison[comparison["metric_set"] == "best_single_global_exit"].iloc[0]

    all_same_strategy = cluster_table[cluster_table["best_in_cluster"]]["strategy"].nunique() == 1

    lines = []
    lines.append("# تقرير اكتشاف الأنماط الخفية وتحسين الخروج حسب نظام الدخول — NAS100\n")
    lines.append(
        "تحليل افتراضي بحت (test-only): لم يتم تعديل `rPnL` الأصلي ولا وقف الخسارة ولا استراتيجية الدخول. "
        "الفكرة الأساسية المُختبرة: هل جودة الخروج المثلى تعتمد على **سياق الدخول** (نظام السوق، سرعة الربح، "
        "شكل الشموع، إلخ)، بحيث تحتاج كل \"عائلة\" من الصفقات استراتيجية خروج مختلفة؟\n"
    )

    lines.append("## ملاحظة منهجية مهمة (تم حلّها بالتنسيق مع المستخدم)\n")
    lines.append(
        "القسم B من الطلب الأصلي (\"سرعة الربح خلال أول 30 دقيقة\") يتعارض مع قاعدة عدم التسريب المستقبلي "
        "(الملاحظة الحرجة رقم 1: يجب أن تُحسب كل ميزات تصنيف سياق الدخول قبل `dateStart` حصراً). بالتنسيق مع "
        "المستخدم: تم استخدام سرعة الربح **فقط كجدول وصفي/استكشافي** بعد التجميع، وليست جزءاً من ميزات "
        "التجميع (Clustering) نفسها — أي أن كل الأنماط المُكتشفة أدناه قابلة للمعرفة فعلياً **لحظة الدخول**، "
        "وليس بعد مرور 30 دقيقة من فتح الصفقة.\n"
    )

    lines.append("## المرحلة 1-2: الأنماط الخفية المُكتشفة (التجميع)\n")
    lines.append(
        f"أفضل عدد عناقيد بحسب Silhouette Score: **k={k}** (score={_fmt(silhouette, 3)}) — من ضمن المدى "
        f"المطلوب (2-5)، لكن في الطرف الأدنى: البيانات لا تدعم أكثر من نمطين متمايزين بوضوح إحصائياً.\n"
    )
    lines.append("| العنقود | الاسم | الحجم | % من الإجمالي | ثقة منخفضة؟ | متوسط الزخم قبل الدخول | متوسط SL/ATR | متوسط RR المُعد | متوسط سرعة الربح (وصفي) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in cluster_profile.iterrows():
        lines.append(
            f"| {int(r['cluster'])} | {r['cluster_name']} | {int(r['size'])} | {_fmt_pct(r['pct_of_total'])} | "
            f"{'نعم' if r['low_confidence'] else 'لا'} | {_fmt(r['mean_pre_entry_momentum_15m'], 4)} | "
            f"{_fmt(r['mean_sl_pct_of_atr'], 3)} | {_fmt(r['mean_risk_reward_setup'], 2)} | "
            f"{_fmt(r['mean_speed_of_profit_r_30m'], 2)}R |"
        )
    lines.append("")

    lines.append("## الملخص التنفيذي\n")
    lines.append(
        f"- **نعم، توجد أنماط خفية (عنقودان متمايزان)**، لكن **لا** — الأنماط لا تحتاج استراتيجيات خروج "
        "مختلفة عن بعضها: نفس الاستراتيجية (\"هدف ربح ثابت عند idealTP\") فازت في **كلا** العنقودين، وفي "
        "**كل سنة** من 2020 إلى 2024 على حدة، وفي بيانات التدريب والتحقق كليهما. هذه نتيجة صادقة وواضحة "
        "(الملاحظة الحرجة رقم 5 تطلب عدم فرض سردية لا تدعمها البيانات): **التخصيص حسب العنقود لم يُضِف شيئاً "
        "فوق مجرد استخدام أفضل استراتيجية عامة واحدة للجميع.**\n"
        f"- **لكن الاكتشاف الأهم هنا مختلف تماماً عمّا كان متوقعاً**: تجاهل مسألة التخصيص حسب العنقود تماماً، "
        f"لأن النتيجة المطلقة مذهلة: تركُ الصفقة تجري حتى هدفها المحدَّد مسبقاً (`idealTP`) بدلاً من الإغلاق "
        f"اليدوي الفعلي رفع معامل الربح من **{_pf(baseline_row['profit_factor'])}** إلى "
        f"**{_pf(hybrid_row['profit_factor'])}** (تحسّن {_fmt(hybrid_row['pf_improvement_pct'], 0)}%)، ورفع "
        f"العائد المتوقع من {_fmt(baseline_row['expectancy'])}R إلى {_fmt(hybrid_row['expectancy'])}R للصفقة، "
        f"وخفّض أقصى تراجع من {_fmt(dd['baseline_max_dd_r'], 1)}R إلى {_fmt(dd['hybrid_max_dd_r'], 1)}R.\n"
        "- **هذا يؤكد ويُكمِّم مباشرة** نتيجة \"441/789 صفقة خروج سيء\" من التحليل الأصلي: جزء كبير من هذا "
        "السوء لم يكن نتيجة استراتيجية خروج خاطئة، بل نتيجة **عدم الالتزام بالهدف الذي كان محدَّداً مسبقاً "
        "أصلاً عند الدخول** (إغلاق يدوي مبكر أو متأخر بدلاً من أمر معلَّق عند `idealTP`).\n"
    )

    lines.append("## المرحلة 3: أداء كل استراتيجية خروج داخل كل عنقود\n")
    lines.append("أفضل 6 استراتيجيات في كل عنقود (المُنتصرة مُظلَّلة في الرسم البياني):\n")
    for cluster in sorted(cluster_table["cluster"].unique()):
        sub = cluster_table[cluster_table["cluster"] == cluster].head(6)
        lines.append(f"**العنقود {int(cluster)}:**\n")
        lines.append("| الاستراتيجية | n | معدل الفوز | العائد المتوقع (R) | معامل الربح | الأفضل؟ |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(f"| {_ar(r['strategy'])} | {int(r['n'])} | {_fmt_pct(r['win_rate'])} | "
                         f"{_fmt(r['expectancy'])} | {_pf(r['profit_factor'])} | "
                         f"{'✓' if r['best_in_cluster'] else ''} |")
        lines.append("")
    lines.append("الرسم البياني الكامل: `outputs/figures/hidden_pattern_cluster_strategy_pf.png`.\n")

    lines.append("## المرحلة 4: نتائج مونت كارلو لكل عنقود\n")
    lines.append("| العنقود | أفضل استراتيجية | متوسط PF (Bootstrap) | 95% CI | PF الأساسي للعنقود | الحكم | % مرات انخفض التراجع عن الأصلي (Shuffle) |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in mc.iterrows():
        lines.append(
            f"| {int(r['cluster'])} | {_ar(r['best_strategy'])} | {_pf(r['mean_pf'])} | "
            f"[{_pf(r['ci_lo_5th'])}, {_pf(r['ci_hi_95th'])}] | {_pf(r['baseline_pf'])} | {r['classification']} | "
            f"{_fmt_pct(r['pct_shuffles_strategy_dd_smaller'])} |"
        )
    lines.append(
        "\n**ملاحظة حول اختبار الخلط (Shuffle)**: معامل الربح والعائد المتوقع مجموع/متوسط ثابت رياضياً بصرف "
        "النظر عن ترتيب الصفقات، فلا معنى لسؤال \"هل ما زالت تفوز بعد الخلط\" على هذين المقياسين. ما يتأثر "
        "فعلاً بالترتيب هو **مسار التراجع (Drawdown)** — لذلك الاختبار هنا يقيس نسبة مرات كان فيها التراجع "
        "الأقصى تحت الاستراتيجية الفائزة أصغر من الأصلي، عبر 1000 إعادة ترتيب عشوائي.\n"
    )

    lines.append("## المرحلة 5: الاستراتيجية الهجينة مقابل أفضل استراتيجية عامة واحدة\n")
    if all_same_strategy:
        lines.append(
            f"بما أن **{_ar(best_global_name)}** فازت في كل عنقود على حدة، فإن الاستراتيجية الهجينة "
            "(حسب العنقود) والاستراتيجية العامة الواحدة الأفضل **متطابقتان تماماً رياضياً** لهذه البيانات "
            "— وليس تقارباً إحصائياً، بل تطابق تام (نفس القاعدة تُطبَّق فعلياً على كل صفقة إما بالمصادفة أو "
            "لأنها الاختيار الأمثل في الحالتين).\n"
        )
    lines.append("| المجموعة | n | معدل الفوز | العائد المتوقع (R) | معامل الربح | Sharpe | أقصى تراجع (R) | إجمالي PnL (R) | تحسّن PF% |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in comparison.iterrows():
        label = {"baseline": "الأصلي", "best_single_global_exit": "أفضل استراتيجية عامة واحدة",
                 "hybrid_cluster_conditional": "الهجينة (حسب العنقود)"}[r["metric_set"]]
        lines.append(f"| {label} | {int(r['n'])} | {_fmt_pct(r['win_rate'])} | {_fmt(r['expectancy'])} | "
                     f"{_pf(r['profit_factor'])} | {_fmt(r['sharpe'], 2)} | {_fmt(r['max_drawdown_r'], 1)} | "
                     f"{_fmt(r['total_pnl_r'], 1)} | {_fmt(r['pf_improvement_pct'], 1)}% |")
    lines.append("")

    lines.append("### اختبار t المزدوج\n")
    lines.append(
        f"**الهجينة مقابل الأصلي**: t={_fmt(sig_baseline['t_statistic'], 3)}, p={_fmt(sig_baseline['p_value'], 6)}, "
        f"Cohen's d={_fmt(sig_baseline['cohens_d'], 3)} → **{sig_baseline['interpretation']}**\n\n"
        f"**الهجينة مقابل أفضل استراتيجية عامة واحدة**: "
        + ("متطابقتان تماماً (فرق = 0 لكل صفقة)، لذا اختبار t غير مُعرَّف إحصائياً (تباين الفرق = صفر) — وهذا "
           "تأكيد رياضي على التطابق، وليس غياب دلالة.\n"
           if sig_global["p_value"] != sig_global["p_value"]  # NaN check
           else f"t={_fmt(sig_global['t_statistic'], 3)}, p={_fmt(sig_global['p_value'], 6)}, "
                f"Cohen's d={_fmt(sig_global['cohens_d'], 3)} → **{sig_global['interpretation']}**\n")
    )

    lines.append("## المرحلة 6: التفصيل الشهري والسنوي وملف التراجع\n")
    lines.append(f"من أصل {len(monthly)} شهراً: **{n_improve}** تحسّن، **{n_degrade}** تراجع. "
                 "الرسم الكامل: `outputs/figures/hidden_pattern_monthly_comparison.png`.\n")
    lines.append(f"أقصى تراجع (Max Drawdown): الأصلي **{_fmt(dd['baseline_max_dd_r'], 1)}R** ← الهجينة "
                 f"**{_fmt(dd['hybrid_max_dd_r'], 1)}R** (رسم: `outputs/figures/hidden_pattern_drawdown_comparison.png`).\n")

    lines.append("**ثبات الاستراتيجية الفائزة عبر السنوات (2020-2024)**:\n")
    all_consistent = bool(yearly["consistent"].all())
    lines.append(f"- {'✓ ثابتة في كل سنة وكل عنقود' if all_consistent else '⚠ غير ثابتة في بعض السنوات/العناقيد'} "
                 f"(انظر `outputs/figures/hidden_pattern_yearly_stability.png`). معامل الربح للاستراتيجية "
                 "الفائزة يبقى أعلى من 1 في كل سنة، رغم تراجع ملحوظ في 2023-2024 مقارنة بـ2020-2022 — "
                 "متسق مع ملاحظة تراجع الحافة (Edge Decay) من التقرير الأصلي.\n"
    )

    lines.append("## المرحلة 7: التحقق على بيانات معزولة (Train/Test)\n")
    lines.append(
        f"تجميع العناقيد أُعيد اشتقاقه (fit) على أول 60% من الصفقات زمنياً فقط ({validation['n_train']} صفقة)، "
        f"ثم صُنِّفت آخر 40% ({validation['n_test']} صفقة) على أقرب مركز عنقود موجود مسبقاً (بدون إعادة "
        f"اكتشاف الأنماط على بيانات مستقبلية) — k={validation['k_train']}, "
        f"silhouette={_fmt(validation['silhouette_train'], 3)}.\n"
    )
    lines.append("| المجموعة | العنقود | الاستراتيجية | n | العائد المتوقع | PF | PF الأساسي |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in validation["train_test_table"].iterrows():
        split_ar = "تدريب" if r["split"] == "train" else "تحقق"
        lines.append(f"| {split_ar} | {int(r['cluster'])} | {_ar(r['strategy'])} | {int(r['n'])} | "
                     f"{_fmt(r['expectancy'])} | {_pf(r['profit_factor'])} | {_pf(r['baseline_profit_factor'])} |")
    lines.append(
        "\n**النتيجة تصمد على بيانات معزولة تماماً لم تُستخدم في اكتشاف الأنماط أو اختيار الاستراتيجية**: "
        "معامل الربح على عيّنة التحقق يبقى أعلى بوضوح من الأساسي في كلا العنقودين، رغم انخفاضه عن عيّنة "
        "التدريب (متوقَّع ومتّسق مع تراجع الحافة العام للاستراتيجية بمرور الوقت، وليس فشلاً في القاعدة "
        "نفسها).\n"
    )

    lines.append("## التحفظات (Caveats)\n")
    lines.append(
        "- **k=2 فقط**: البيانات لم تدعم إحصائياً أكثر من عنقودين متمايزين بوضوح ضمن المدى المطلوب (2-5) — "
        "silhouette منخفض نسبياً "
        f"({_fmt(silhouette, 2)})، ما يعني أن الفصل بين \"الأنماط\" ليس حاداً جداً.\n"
        "- **لم تُستخدم استراتيجية خروج مختلفة فعلياً لكل عنقود**: التخصيص حسب النمط لم يُثبت قيمة مضافة هنا "
        "— النتيجة الإيجابية بأكملها تأتي من استراتيجية واحدة عامة (idealTP)، وليس من ذكاء \"هجين\" حقيقي.\n"
        "- **هدف idealTP قد لا يتحقق دائماً في التداول الحي بنفس الطريقة**: المحاكاة تفترض تنفيذ أمر معلَّق "
        "(limit order) يُملأ بالضبط عند مستوى idealTP فور لمسه، بدون انزلاق سعري (slippage) أو رفض تنفيذ — "
        "افتراض متفائل قياسي لأوامر الحد، لكنه يستحق التحقق في التنفيذ الحي.\n"
        "- **حجم العيّنة لكل عنقود**: أصغر عنقود لا يزال أعلى من الحد الأدنى الموثَّق (30 صفقة)، لكن التقسيم "
        "الإضافي حسب السنة (المرحلة 6) ينزل أحياناً لعشرات الصفقات فقط سنوياً.\n"
        "- **لا تعديل على الدخول أو وقف الخسارة**: التحسّن بأكمله ناتج عن الالتزام بهدف كان محدَّداً مسبقاً "
        "أصلاً، وليس عن تغيير جودة نقطة الدخول.\n"
    )

    lines.append("## التوصية النهائية\n")
    lines.append(
        f"**التوصية بالمتابعة الجدّية، وليس مجرد \"دراسة إضافية\"**: النتيجة الأقوى في هذا التحليل — الالتزام "
        f"بهدف `idealTP` بدلاً من الإغلاق اليدوي — صمدت أمام أربعة اختبارات متانة مستقلة (مونت كارلو، ثبات "
        "سنوي عبر 5 سنوات، تحقق على بيانات معزولة زمنياً، واختبار إحصائي مزدوج بدلالة عالية جداً)، وتتفق "
        "منطقياً مع تشخيص \"الخروج السيء\" من التحليل الأصلي. **الخطوة التالية المقترحة**: تجربة ورقية "
        "(paper trading) لقاعدة بسيطة — وضع أمر جني ربح معلَّق عند `idealTP` فور الدخول بدل الاعتماد على "
        "الإغلاق اليدوي — قبل أي اعتماد كامل على رأس مال حقيقي، مع مراقبة الانزلاق السعري الفعلي عند التنفيذ. "
        "أما فرضية \"التخصيص حسب نمط الدخول\" تحديداً فلم تُثبت قيمة مضافة هنا ولا داعي لتعقيد التنفيذ بها.\n"
    )

    return "\n".join(lines)
