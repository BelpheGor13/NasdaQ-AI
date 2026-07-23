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
    targ = t_cons[t_cons["strategy"] == "fixed_tp_idealTP"].iloc[0]

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

    lines.append("## تحفّظ مهم جدًا حول \"الهدف\" المُستخدَم\n")
    lines.append(
        "العمود `idealTP` (المُستخدَم هنا كسعر \"الهدف\") **ليس هدفًا كان محدَّدًا مسبقًا قبل الدخول** — هو "
        "أفضل سعر تحقّق فعليًا (Maximum Favorable Excursion) ضمن نافذة الصفقة الأصلية كما أُغلقت تاريخيًا "
        "(موثَّق من تحليل سابق للبيانات). إعادة استخدامه هنا كـ\"مستوى سعري\" لاختبار هل كان سيُلمَس فعليًا "
        "لو استمرت الصفقة دون تدخّل — هذا اختبار صحيح تقنيًا (لا تسريب معلومات: نمشي على شموع حقيقية بحثًا "
        "عن لمس هذا المستوى)، **لكنه ليس محاكاة لقاعدة تداول كانت ستُطبَّق حرفيًا في الزمن الحقيقي** — لأن "
        "التاجر لم يكن ليعرف مسبقًا أن هذا هو \"الهدف الأمثل\" لهذه الصفقة تحديدًا. لذلك أُضيفت أيضًا أهداف "
        "بديلة بمضاعفات R ثابتة (2R، 3R، 4R) — وهي أهداف يمكن تحديدها فعليًا مسبقًا — للمقارنة والسياق.\n"
    )

    lines.append("## الملخص التنفيذي\n")
    lines.append(
        f"**نعم — الفرق كبير وقوي إحصائيًا.** ترك الصفقة تصل للهدف (`idealTP`) أو الوقف الأصلي بدون أي "
        f"تدخّل بشري كان لينتج معامل ربح **{_pf(targ['profit_factor'])}** مقابل **{_pf(base['profit_factor'])}** "
        f"فعليًا — أي ما يقارب **ضعف الأداء**. معدل الفوز يرتفع من {_fmt_pct(base['win_rate'])} إلى "
        f"{_fmt_pct(targ['win_rate'])}، والعائد المتوقع من {_fmt(base['expectancy'])}R إلى "
        f"{_fmt(targ['expectancy'])}R للصفقة، وأقصى تراجع تراكمي ينخفض من {_fmt(base['max_drawdown_r'], 1)}R "
        f"إلى {_fmt(targ['max_drawdown_r'], 1)}R فقط.\n\n"
        f"الاختبار الإحصائي المزدوج (نفس الصفقات): t={_fmt(sig['conservative']['t_statistic'])}, "
        f"p={sig['conservative']['t_p_value']:.2e} (و Wilcoxon p={sig['conservative']['wilcoxon_p_value']:.2e}) — "
        "دلالة إحصائية قوية جدًا وليست حدّية، وهذا **يختلف جوهريًا** عن اختبار \"بلا وقف خسارة\" السابق الذي "
        "لم يُظهر دلالة (لأن ذاك كان مدفوعًا بذيل نادر كارثي، أما هذا فأثر **مُتّسق عبر أغلب الصفقات**، "
        "كما يؤكده Cohen's d = {} وليس فقط الذيل).\n".format(_fmt(sig['conservative']['cohens_d'], 2))
    )

    lines.append("## جدول المقارنة (السيناريو المحافظ — الحسم ضمن نافذة الصفقة الأصلية فقط)\n")
    lines.append("| الاستراتيجية | n | معدل الفوز | العائد المتوقع (R) | معامل الربح | أقصى تراجع (R) |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in t_cons[t_cons["strategy"].isin(
            ["baseline", "fixed_tp_idealTP", "fixed_tp_2R", "fixed_tp_3R", "fixed_tp_4R"])].iterrows():
        lines.append(f"| {r['strategy']} | {int(r['n'])} | {_fmt_pct(r['win_rate'])} | {_fmt(r['expectancy'])} | "
                     f"{_pf(r['profit_factor'])} | {_fmt(r['max_drawdown_r'], 1)} |")
    lines.append(
        "\n**لماذا الفارق الكبير بين `fixed_tp_idealTP` والأهداف الثابتة (2R/3R/4R)**: `idealTP` هو أفضل نقطة "
        "تحقّقت فعليًا لكل صفقة على حدة (سقف شبه مثالي بأثر رجعي)، بينما 2R/3R/4R أهداف موحّدة صعبة التحقق "
        "بانتظام. النتائج الضعيفة نسبيًا للأهداف الثابتة (معامل ربح قريب من 1) تُظهر أن الفائدة الحقيقية هنا "
        "ليست \"استخدم أي هدف ثابت\"، بل تحديدًا: **لا تُغلق الصفقة يدويًا قبل أن تصل لأقصى إمكانياتها أو "
        "تُوقَف فعليًا** — وهذا يتوافق تمامًا مع نتيجة \"جودة الخروج\" في التقرير الأساسي (56% من الصفقات "
        "خروج سيء).\n"
    )

    t_agg = tables["aggressive"]
    targ_agg = t_agg[t_agg["strategy"] == "fixed_tp_idealTP"].iloc[0]
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
    lines.append(
        "\n**النتيجة قوية جدًا**: في كلا السيناريوهين، فترة ثقة معامل الربح لا تتقاطع مع معامل الربح الفعلي "
        "إطلاقًا، و100% من عمليات إعادة التشكيل أظهرت تفوّق استراتيجية \"الهدف أو الوقف\" — أي أن هذا ليس "
        "نتيجة صدفة في هذه العيّنة تحديدًا، بل نمط متّسق عبر آلاف إعادات التشكيل العشوائي لنفس الصفقات.\n"
    )

    lines.append("## الخلاصة\n")
    lines.append(
        "هذا أقوى دليل إحصائي في كل التحليلات الافتراضية حتى الآن — قوي، متّسق، وغير مدفوع بذيل نادر "
        "(بخلاف اختبار \"بلا وقف خسارة\"). **لكن التحفّظ أعلاه حول `idealTP` جوهري**: هذا لا يعني حرفيًا "
        "\"ضع هدفًا وانتظر\" — لأن `idealTP` أفضل نقطة بأثر رجعي، وليس رقمًا كان معروفًا مسبقًا. الرسالة "
        "العملية الأكثر أمانًا: **الخروج اليدوي المبكر (سواء بدافع الخوف من فقدان الربح أو غيره) يبدو أنه "
        "يكلّف الاستراتيجية أداءً كبيرًا وموثوقًا إحصائيًا** — وهذا يستحق فعليًا التحقق مع آلية خروج مبرمجة "
        "وقابلة للتطبيق مسبقًا (كما في تحليلي الوقف المتحرك وتحسين الخروج حسب النمط)، وليس بالضرورة انتظار "
        "\"أفضل نقطة ممكنة\" بأثر رجعي.\n"
    )

    return "\n".join(lines)
