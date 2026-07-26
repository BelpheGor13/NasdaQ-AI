"""Arabic report for the end-of-day (no overnight holding) target-or-stop
rule -- direct answer to a user-specified rule: let the trade run to its
target or original stop, but force-close at market at 16:00 America/
New_York the same day if neither is hit, never holding overnight.
"""
from src import config


def _fmt(x, digits=3):
    return f"{x:.{digits}f}" if x == x else "غير متوفر"


def _fmt_pct(x, digits=1):
    return f"{x*100:.{digits}f}%" if x == x else "غير متوفر"


def _pf(x):
    return "∞" if x == float("inf") else _fmt(x, 3)


def build_report(results: dict) -> str:
    s = results["summary"]
    sig = results["significance"]
    mc = results["monte_carlo"]
    reasons = results["exit_reasons"]
    deadline_hour = results["deadline_hour_ny"]

    lines = []
    lines.append(f"# تقرير: الهدف أو الوقف بدون تبييت (إغلاق إجباري الساعة {deadline_hour}:00 نيويورك) — NAS100\n")
    lines.append(
        f"قاعدة محدَّدة من المستخدم: اترك الصفقة تمشي لهدفها الحقيقي (`maxTP` لو موجود، وإلا ضعف مسافة وقف "
        f"الخسارة 2R كحد أدنى افتراضي — `src/target_resolution.py`) أو وقف خسارتها الأصلي (`initalSL`)، "
        f"لكن **أغلقها إجباريًا بسعر السوق الساعة {deadline_hour}:00 بتوقيت نيويورك من نفس يوم الدخول** "
        "إذا لم يتحقق أي منهما بعد — بدون أي تبييت لليوم التالي إطلاقًا. اختُبرت على نفس "
        f"{s['n']} صفقة، بدون تعديل أي بيانات أصلية.\n"
    )

    lines.append("## ⚠️ تصحيح منهجي حول تعريف \"الهدف\" (نسخة سابقة استخدمت `idealTP` خطأً)\n")
    lines.append(
        "نسخة سابقة من هذا التقرير استخدمت عمود `idealTP` مباشرة كـ\"الهدف\" (PF 2.42 آنذاك)، بنفس الخطأ "
        "الذي وقع فيه تقرير \"الهدف أو الوقف\" الأصلي بلا حد زمني. **`idealTP` ليس هدفاً — هو أعلى سعر "
        "وصلته الصفقة قبل ضرب وقف الخسارة الأصلي (MFE قبل الستوب)**، وهو مرتفع بشكل مصطنع مقارنة بأي هدف "
        "حقيقي (تفاصيل كاملة في `src/idealtp_data_quality_check.py`). كل الأرقام أدناه الآن مبنية على "
        "**الهدف الحقيقي** (`src/target_resolution.py`): `maxTP` لما يكون موجوداً (213 صفقة فقط)، وإلا "
        "2R كحد أدنى افتراضي لبقية الـ576 صفقة.\n"
    )

    lines.append("## الملخص التنفيذي\n")
    lines.append(
        f"**النتيجة لسا أفضل من الفعلي، بأرقام أهدأ وأكثر واقعية بعد التصحيح** — معامل الربح "
        f"**{_pf(s['pf_eod'])}** مقابل **{_pf(s['pf_orig'])}** فعليًا (مقارنةً بـ1.304 لنسخة \"بلا حد "
        "زمني\" المصحَّحة من التقرير الآخر — أي أن الإغلاق الإجباري في نفس اليوم يضيف تحسّناً إضافياً "
        f"متواضعاً، مو مضاعفة الأداء). معدل الفوز يرتفع من "
        f"{_fmt_pct(s['win_rate_orig'])} إلى {_fmt_pct(s['win_rate_eod'])}، وأقصى تراجع تراكمي ينخفض من "
        f"{_fmt(s['max_dd_orig'], 1)}R إلى **{_fmt(s['max_dd_eod'], 1)}R فقط** — أقل من نسخة \"بلا حد "
        "زمني\" المصحَّحة (32.6R)، لأن الإغلاق الإجباري في نفس اليوم يمنع الصفقات الرابحة من الانعكاس لاحقًا "
        "وضرب الوقف بعد أيام.\n\n"
        f"الاختبار الإحصائي المزدوج: t={_fmt(sig['t_statistic'])}, p={sig['t_p_value']:.2e} "
        f"(Wilcoxon p={sig['wilcoxon_p_value']:.2e})، Cohen's d={_fmt(sig['cohens_d'], 2)} — دلالة إحصائية "
        "موجودة لكن حجم الأثر صغير (d قريب من 0.1)، متّسق مع بقية تحليلات \"الهدف أو الوقف\" بعد التصحيح.\n"
    )

    lines.append("## جدول المقارنة\n")
    lines.append("| القاعدة | n | متوسط (R) | وسيط (R) | معدل الفوز | معامل الربح | أقصى تراجع (R) |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| الفعلي (إغلاق يدوي) | {s['n']} | {_fmt(s['mean_orig'])} | {_fmt(s['median_orig'])} | "
                 f"{_fmt_pct(s['win_rate_orig'])} | {_pf(s['pf_orig'])} | {_fmt(s['max_dd_orig'], 1)} |")
    lines.append(f"| هدف أو وقف + إغلاق إجباري {deadline_hour}:00 | {s['n']} | {_fmt(s['mean_eod'])} | "
                 f"{_fmt(s['median_eod'])} | {_fmt_pct(s['win_rate_eod'])} | {_pf(s['pf_eod'])} | "
                 f"{_fmt(s['max_dd_eod'], 1)} |")
    lines.append("")

    lines.append("## كيف حُسمت الصفقات\n")
    lines.append("| السبب | عدد الصفقات | النسبة |")
    lines.append("|---|---|---|")
    labels = {"target": "لمست الهدف قبل الموعد", "stop": "لمست الوقف الأصلي قبل الموعد",
              "eod_forced": f"لم تحسم — أُغلقت إجباريًا الساعة {deadline_hour}:00"}
    for reason, n in reasons.items():
        lines.append(f"| {labels.get(reason, reason)} | {n} | {_fmt_pct(n/s['n'])} |")
    lines.append(
        f"\nفقط **{_fmt_pct(reasons.get('eod_forced', 0)/s['n'])}** من الصفقات احتاجت فعليًا للإغلاق "
        "الإجباري — الغالبية العظمى تُحسم طبيعيًا (هدف أو وقف) قبل نهاية اليوم أصلاً.\n"
    )

    lines.append("## التحقق بمونت كارلو (Bootstrap مزدوج)\n")
    lines.append(
        f"- متوسط PF عبر إعادة التشكيل: {_pf(mc['mean_pf'])} (فترة ثقة 90%: {_pf(mc['ci_lo_5th'])} – {_pf(mc['ci_hi_95th'])})\n"
        f"- PF الفعلي: {_pf(mc['observed_baseline_pf'])} — {'خارج' if not mc['ci_crosses_baseline'] else 'داخل'} فترة الثقة\n"
        f"- نسبة مرات تفوّق القاعدة الجديدة على الفعلي: **{_fmt_pct(mc['pct_boot_iters_beats_baseline'])}**\n"
    )

    lines.append("## تحفّظات\n")
    lines.append(
        "- **576 من أصل 789 صفقة تستخدم افتراض 2R** (ضعف مسافة وقف الخسارة) كهدف، لأن هدفها الحقيقي غير "
        "مسجَّل (لم تصله أصلاً) — فقط 213 صفقة تستخدم `maxTP` الحقيقي.\n"
        "- **تعارض نادر بنفس الشمعة**: نفس افتراض ترجيح الهدف عند تعارض الوقف والهدف بنفس دقيقة الشمعة "
        "(أثره مقاس سابقًا بأقل من 2.4% على رقم مشابه).\n"
        "- **حساسية موعد الإغلاق**: النتائج مبنية على 16:00 نيويورك تحديدًا؛ لم تُختبر حساسية النتيجة لتغيير "
        "هذا الموعد ساعة أو ساعتين — يستحق فحصًا إضافيًا قبل الاعتماد الكامل.\n"
        "- **الصفقة الأكبر أثرًا** (~36R في يوم واحد) صفقة حقيقية بوقف ضيق جدًا صادفت تحركًا سعريًا كبيرًا "
        "نفس اليوم — نتيجة صحيحة وليست خطأ حسابيًا، لكنها تُذكّر أن جزءًا من الأداء القوي هنا يعتمد على عدد "
        "قليل من الأيام شديدة التقلب.\n"
    )

    lines.append("## الخلاصة\n")
    lines.append(
        f"هذه القاعدة (هدف أو وقف + إغلاق إجباري {deadline_hour}:00 نيويورك، بلا تبييت مطلقًا) تتحسّن قليلاً "
        "عن نسخة \"بلا حد زمني\" المصحَّحة على المقاييس الثلاثة معًا (معامل الربح، معدل الفوز، أقصى تراجع) — "
        "وتحافظ على الانضباط العملي بعدم حمل مخاطرة ليلية. **لكن الفارق بينها وبين النسخة بلا حد زمني متواضع** "
        "بعد التصحيح (مو \"الأقوى بفارق كبير\" كما ادّعت النسخة السابقة المبنية على `idealTP`) — يستاهل "
        "فحصاً إضافياً مع آلية خروج مبرمجة، لا اعتماداً فورياً كقاعدة نهائية.\n"
    )

    return "\n".join(lines)
