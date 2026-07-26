"""Assembles the final Arabic report for the target-or-stop scenario test
(direct answer to: "what if the trade were just left to hit its target or
its original stop, no early manual exit, no trailing?"). Code/columns stay
English; narrative is Arabic, same convention as the rest of this project.
"""
from src import config


def _fmt(x, digits=3):
    return f"{x:.{digits}f}" if x == x else "غير متوفر"


def _fmt_pct(x, digits=1):
    return f"{x*100:.{digits}f}%" if x == x else "غير متوفر"


def _pf(x):
    return "∞" if x == float("inf") else _fmt(x, 3)


def build_report(results: dict) -> str:
    tables = results["tables"]  # {"conservative": df, "aggressive": df}
    sig = results["significance"]  # {"conservative": dict, "aggressive": dict}
    mc = results["monte_carlo"]  # {"conservative": dict, "aggressive": dict}
    pct_unresolved = results["pct_unresolved"]  # {"conservative": float, "aggressive": float}

    t_cons = tables["conservative"]
    base = t_cons[t_cons["strategy"] == "baseline"].iloc[0]
    targ = t_cons[t_cons["strategy"] == "fixed_tp_real_target"].iloc[0]

    lines = []
    lines.append("# تقرير: هل ترك الصفقة تصل للهدف أو الوقف بدون تدخّل يُحسِّن الأداء؟ — NAS100\n")
    lines.append(
        "إجابة مباشرة على سؤال محدد: لو تُركت كل صفقة بسعر الدخول ووقف الخسارة الأصليين كما هما، بدون أي "
        "خروج يدوي مبكر وبدون تحريك الوقف، حتى تلمس السعر إما الهدف أو الوقف فعليًا — كيف كانت النتيجة "
        "لتختلف عن الخروج الفعلي (المتضمّن قرارات الوسيط/التاجر)؟ هذا تحليل افتراضي بحت لا يُعدِّل السجل "
        f"الأصلي، على نفس الـ{int(base['n'])} صفقة. يُعيد استخدام محرك المحاكاة المبني مسبقًا "
        "(`exit_strategy_simulation.py`) الذي يُبقي وقف الخسارة الأصلي فعّالًا طوال الوقت في كل استراتيجية "
        "بديلة، ولا يضيف أي محاكاة جديدة — فقط يستخرج ويُبرز هذه المقارنة المحدَّدة بوضوح.\n"
    )

    lines.append("## ⚠️ تصحيح منهجي حول تعريف \"الهدف\" (نسخة سابقة استخدمت `idealTP` خطأً)\n")
    lines.append(
        "نسخة سابقة من هذا التقرير استخدمت عمود `idealTP` مباشرة كـ\"الهدف\"، مع تحفّظ بأنه \"أفضل سعر "
        "تحقّق بأثر رجعي\" — لكن ظلّت تستخدمه فعليًا كمستوى الهدف في كل الأرقام. هذا كان خطأً أعمق من مجرد "
        "تحفّظ: **`idealTP` ليس هدفاً على الإطلاق، حتى بأثر رجعي — هو أعلى سعر وصلته الصفقة قبل ضرب وقف "
        "الخسارة الأصلي (MFE قبل الستوب)**، وهو مرتفع بشكل مصطنع مقارنة بأي هدف كان يمكن تحديده فعلاً وقت "
        "الدخول (تحقّق من هذا التعريف بمطابقته ضد حساب مستقل من الشموع، تفاصيل كاملة في "
        "`src/idealtp_data_quality_check.py`).\n\n"
        "**كل الأرقام في هذا التقرير الآن مبنية على الهدف الحقيقي** (`src/target_resolution.py`، الوحدة "
        "المرجعية الموحّدة بكل ملفات المشروع): عمود `maxTP` لما يكون موجوداً (213 صفقة فقط أُغلقت بربح "
        "حقيقي)، وإلا **ضعف مسافة وقف الخسارة (2R) كحد أدنى افتراضي** لبقية الـ576 صفقة — حسب قاعدة "
        "المستخدم نفسه، لأن الهدف الحقيقي غير مسجَّل لصفقة لم تصله أصلاً. الأهداف الثابتة (2R، 3R، 4R) "
        "لسا مُبقاة بالجدول تحت للمقارنة والسياق.\n"
    )

    lines.append("## الملخص التنفيذي\n")
    lines.append(
        f"**نعم — لا يزال هناك فرق، بأرقام أهدأ وأكثر واقعية بعد التصحيح.** ترك الصفقة تصل لهدفها الحقيقي "
        f"(`maxTP`، أو 2R كحد أدنى لو غير معروف) أو الوقف الأصلي بدون أي تدخّل بشري كان لينتج معامل ربح "
        f"**{_pf(targ['profit_factor'])}** مقابل **{_pf(base['profit_factor'])}** "
        f"فعليًا. معدل الفوز يرتفع من {_fmt_pct(base['win_rate'])} إلى "
        f"{_fmt_pct(targ['win_rate'])}، والعائد المتوقع من {_fmt(base['expectancy'])}R إلى "
        f"{_fmt(targ['expectancy'])}R للصفقة، وأقصى تراجع تراكمي ينخفض من {_fmt(base['max_drawdown_r'], 1)}R "
        f"إلى {_fmt(targ['max_drawdown_r'], 1)}R فقط.\n\n"
    )
    t_p = sig['conservative']['t_p_value']
    w_p = sig['conservative']['wilcoxon_p_value']
    both_sig = t_p < 0.05 and w_p < 0.05
    sig_note = ("دلالة إحصائية بكلا الاختبارين (t وWilcoxon)" if both_sig else
                "دلالة إحصائية بحسب اختبار t، لكن Wilcoxon أضعف/غير حاسم (t وWilcoxon لا يتفقان دائماً "
                "بحجم أثر صغير كهذا)")
    lines.append(
        f"الاختبار الإحصائي المزدوج (نفس الصفقات): t={_fmt(sig['conservative']['t_statistic'])}, "
        f"p={t_p:.2e} (Wilcoxon p={w_p:.2e}) — {sig_note}. **بصراحة**: Cohen's d = "
        f"{_fmt(sig['conservative']['cohens_d'], 2)} هو حجم أثر صغير جداً — الفرق حقيقي ومتّسق إحصائياً "
        "عبر آلاف عمليات إعادة التشكيل (شوف قسم مونت كارلو تحت)، لكنه ليس فرقاً دراماتيكياً على مستوى "
        "الصفقة الواحدة.\n"
    )

    lines.append("## جدول المقارنة (السيناريو المحافظ — الحسم ضمن نافذة الصفقة الأصلية فقط)\n")
    lines.append("| الاستراتيجية | n | معدل الفوز | العائد المتوقع (R) | معامل الربح | أقصى تراجع (R) |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in t_cons[t_cons["strategy"].isin(
            ["baseline", "fixed_tp_real_target", "fixed_tp_2R", "fixed_tp_3R", "fixed_tp_4R"])].iterrows():
        lines.append(f"| {r['strategy']} | {int(r['n'])} | {_fmt_pct(r['win_rate'])} | {_fmt(r['expectancy'])} | "
                     f"{_pf(r['profit_factor'])} | {_fmt(r['max_drawdown_r'], 1)} |")
    lines.append(
        "\n**الفارق بين `fixed_tp_real_target` والأهداف الثابتة (2R/3R/4R)**: `fixed_tp_real_target` يستخدم "
        "هدف كل صفقة الخاص بها (`maxTP` الحقيقي لما يكون موجوداً، وإلا 2R كحد أدنى) بدل هدف موحّد للجميع — "
        "وهذا يفسّر أي فارق أداء بين الصفوف أعلاه.\n"
    )

    t_agg = tables["aggressive"]
    targ_agg = t_agg[t_agg["strategy"] == "fixed_tp_real_target"].iloc[0]
    lines.append("## السيناريو المتشدد (تمديد 3 أيام بحثًا عن الحسم)\n")
    lines.append(
        f"معامل الربح تحت التمديد: {_pf(targ_agg['profit_factor'])} (مقابل {_pf(targ['profit_factor'])} في "
        f"السيناريو المحافظ) — **النتيجة لا تتغير جوهريًا**. نسبة الصفقات غير المحسومة "
        f"(لم تلمس الهدف أو الوقف حتى نهاية النافذة): {_fmt_pct(pct_unresolved['conservative'])} في السيناريو "
        f"المحافظ، و{_fmt_pct(pct_unresolved['aggressive'])} فقط في المتشدد — أي أن الغالبية العظمى من "
        "الصفقات تُحسم ضمن نافذتها الزمنية الأصلية أصلاً، وهذا ليس افتراضًا بعيد المدى.\n"
    )

    lines.append("## التحقق بمونت كارلو (Bootstrap مزدوج، 2000 إعادة تشكيل)\n")
    lines.append("| السيناريو | متوسط PF (Bootstrap) | الحد الأدنى 5% | الحد الأعلى 95% | PF الفعلي | يتقاطع؟ | % مرات يتفوق |")
    lines.append("|---|---|---|---|---|---|---|")
    for scenario in ("conservative", "aggressive"):
        m = mc[scenario]
        lines.append(f"| {scenario} | {_pf(m['mean_pf'])} | {_pf(m['ci_lo_5th'])} | {_pf(m['ci_hi_95th'])} | "
                     f"{_pf(m['observed_baseline_pf'])} | {'نعم' if m['ci_crosses_baseline'] else 'لا'} | "
                     f"{_fmt_pct(m['pct_boot_iters_beats_baseline'])} |")
    mc_cons_pct = mc['conservative']['pct_boot_iters_beats_baseline']
    mc_agg_pct = mc['aggressive']['pct_boot_iters_beats_baseline']
    lines.append(
        "\n**نتيجة متّسقة عبر إعادة التشكيل**: في كلا السيناريوهين، فترة ثقة 90% لمعامل الربح لا تتقاطع مع "
        f"معامل الربح الفعلي، و{_fmt_pct(mc_cons_pct)} (محافظ) / {_fmt_pct(mc_agg_pct)} (متشدد) من عمليات "
        "إعادة التشكيل (Bootstrap) أظهرت تفوّق استراتيجية \"الهدف أو الوقف\" — أي أن هذا ليس نتيجة صدفة "
        "في هذه العيّنة تحديدًا، بل نمط متّسق عبر آلاف إعادات التشكيل العشوائي لنفس الصفقات.\n"
    )

    lines.append("## الخلاصة\n")
    lines.append(
        "بعد التصحيح، النتيجة أهدأ من الادعاء السابق (\"أقوى دليل إحصائي في كل التحليلات\") لكنها لسا "
        "بنفس الاتجاه: ترك الصفقة تصل لهدفها الحقيقي أو وقفها الأصلي، بدون خروج يدوي مبكر، ينتج أداءً أفضل "
        "من الخروج الفعلي على هذه العيّنة. **الرسالة العملية**: هذا لا يعني حرفيًا \"ضع هدفًا وانتظر\" بلا "
        "قاعدة مبرمجة — 576 من أصل 789 صفقة تستخدم هنا افتراض 2R لأن هدفها الحقيقي غير مسجَّل أصلاً (لم "
        "تصله). النتيجة تستاهل فحصاً إضافياً مع آلية خروج مبرمجة وقابلة للتطبيق مسبقًا (كما في تحليلي الوقف "
        "المتحرك وتحسين الخروج حسب النمط)، وليس اعتمادها كما هي كقاعدة نهائية.\n"
    )

    return "\n".join(lines)
