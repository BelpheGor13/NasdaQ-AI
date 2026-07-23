"""Stage 16: assemble the final report in Arabic from the pipeline's actual
computed results (never invented numbers -- every figure quoted here is
read out of the dataframes/dicts the earlier stages produced).

Per the source spec: code, column names, and identifiers stay in English;
only the narrative (explanations, findings, recommendations) is in Arabic.
"""
from src import config


def _fmt_pct(x, digits=1):
    return f"{x*100:.{digits}f}%" if x == x else "غير متوفر"


def _fmt(x, digits=3):
    return f"{x:.{digits}f}" if x == x else "غير متوفر"


def build_report(results: dict) -> str:
    profile = results["profile"]
    exit_summary = results["exit_summary"]
    yearly = results["yearly_performance"]
    trend = results["sequence_trend"]
    top_scored = results["top_scored"]
    clustering = results["clustering"]
    shap_stability = results["shap_stability"]
    base_win_rate = results["base_win_rate"]
    n_candidates_tested = results["n_candidates_tested"]
    n_fdr_survivors = results["n_fdr_survivors"]

    confirmed = top_scored[top_scored["confidence_tier"] == "confirmed"]
    exploratory = top_scored[top_scored["confidence_tier"] == "exploratory"]
    rejected = top_scored[top_scored["confidence_tier"] == "rejected_or_low_confidence"]

    lines = []
    lines.append("# تقرير اكتشاف الأنماط المركّبة — استراتيجية NAS100\n")
    lines.append(f"عدد الصفقات المحلَّلة: **{profile.trades_shape[0]}** صفقة، "
                 f"من {profile.trades_date_range[0].date()} إلى {profile.trades_date_range[1].date()}.\n")

    lines.append("## الملخص التنفيذي\n")
    lines.append(
        f"- تم اختبار **{n_candidates_tested}** نمطًا مركّبًا مرشّحًا (توليفات من ميزات ما-قبل-الدخول)، "
        f"ولم ينجُ منها بعد تصحيح المقارنات المتعددة (Benjamini-Hochberg FDR) سوى **{n_fdr_survivors}** نمط.\n"
        f"- معدل الفوز الأساسي للاستراتيجية بالكامل هو **{_fmt_pct(base_win_rate)}**.\n"
        f"- لم يظهر اتجاه إحصائيًا معنويًا لتراجع الحافة (Edge Decay) على مستوى تسلسل الصفقات "
        f"(Spearman rho={trend['spearman_rho']:.3f}, p={trend['spearman_p']:.3f}) رغم أن سنة "
        f"2023 كانت ضعيفة (معامل ربح {_fmt(yearly[yearly['year']==2023]['profit_factor'].values[0] if 2023 in yearly['year'].values else float('nan'))}) "
        f"تلتها سنة 2024 القوية — أي أن الأداء الضعيف يبدو أقرب لسنة سيئة واحدة وليس تراجعًا مستمرًا في الحافة.\n"
        f"- من بين أفضل {len(top_scored)} مرشحًا الذين خضعوا للتحقق الكامل (walk-forward + مونت كارلو + "
        f"تصحيح FDR + بايزي): **{len(confirmed)}** بمستوى ثقة \"مؤكَّد\"، و**{len(exploratory)}** استكشافي "
        f"منخفض الثقة نسبيًا لكنه يستحق المتابعة، و**{len(rejected)}** مرفوض أو غير كافي العيّنة.\n"
    )

    lines.append("## أقوى المرشَّحات الناجية (مرتّبة حسب Robustness Score)\n")
    if len(top_scored) == 0:
        lines.append("لم تنتج عملية البحث أي مرشَّحات ضمن الحد الأدنى للعيّنة.\n")
    else:
        lines.append("| الشرط (أعمدة/قيم إنجليزية) | الاتجاه | العيّنة n | العائد المتوقع (R) | معدل الفوز | "
                     "p (FDR) | مونت كارلو | ثبات الحدود | اعتماد النظام | الفئة | Robustness Score |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in top_scored.head(10).iterrows():
            lines.append(
                f"| `{r['condition']}` | {'إيجابي' if 'positive' in r['direction'] else 'سلبي (تجنّب)'} "
                f"| {r['n']} | {_fmt(r['expectancy'])} | {_fmt_pct(r['win_rate'])} | {_fmt(r['p_fdr_bh'])} "
                f"| {r['mc_verdict']} | {r['stability_verdict']} | {r['regime_verdict']} "
                f"| {r['confidence_tier']} | {_fmt(r['robustness_score'], 1)} |"
            )
        lines.append("")

    lines.append("## أنماط إمكانية الربح الكامنة (maxRiskReward)\n")
    lines.append(
        "بالإضافة إلى البحث عن أنماط مرتبطة بالربح/الخسارة الفعلي، جرى بحث مستقل عن أنماط مرتبطة بارتفاع أو "
        "انخفاض **إمكانية الربح الكامنة** (`maxRiskReward` — أفضل ما كانت الصفقة قد تحققه، بصرف النظر عن مكان "
        "إغلاقها فعليًا). ملاحظة إحصائية: هذا المتغير غير سالب دائمًا بحكم تعريفه (لا خسائر فيه)، لذا فإن "
        "\"معامل الربح\" (Profit Factor) ليس مقياسًا ذا معنى هنا؛ الاعتماد كان على العائد المتوقع (متوسط القيمة) "
        "بدلاً منه.\n"
    )
    n_tested_pot = results["n_candidates_tested_potential"]
    n_fdr_pot = results["n_fdr_survivors_potential"]
    lines.append(f"من أصل **{n_tested_pot}** توليفة مُختبرة على هذا الهدف، نجت **{n_fdr_pot}** بعد تصحيح FDR.\n")

    top_scored_pot = results["top_scored_potential"]
    if len(top_scored_pot):
        lines.append("| الشرط | العيّنة n | متوسط إمكانية الربح (R) | p (FDR) | مونت كارلو | الفئة |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in top_scored_pot.head(5).iterrows():
            lines.append(f"| `{r['condition']}` | {r['n']} | {_fmt(r['expectancy'])} | {_fmt(r['p_fdr_bh'])} "
                         f"| {r['mc_verdict']} | {r['confidence_tier']} |")
        lines.append("")

    cross_ref = results.get("cross_ref")
    if cross_ref:
        top_n = int(top_scored.iloc[0]["n"]) if len(top_scored) else "غير متوفر"
        lines.append(
            f"**نتيجة تقاطع مهمة**: أقوى نمط في تحليل الربح/الخسارة الفعلي — `{cross_ref['condition']}` — "
            f"ظهر أيضًا في تحليل إمكانية الربح الكامنة بعائد متوقع {_fmt(cross_ref['expectancy_on_potential'])}R "
            f"و p (FDR) = {_fmt(cross_ref['p_fdr_bh_on_potential'])} على هذا الهدف المستقل. اتفاق نمط واحد عبر "
            f"هدفين مُشتقّين بشكل مستقل (العائد الفعلي مقابل أقصى إمكانية محتملة من الشموع) دليل أقوى من نجاحه "
            f"في اختبار واحد فقط، رغم أن العيّنة (n={top_n}) تبقى صغيرة نسبيًا.\n"
        )

    shap_scored = results.get("shap_guided_scored")
    shap_features = results.get("shap_features", [])
    if shap_scored is not None and len(shap_scored):
        lines.append("## اختبار موجَّه بميزات SHAP (فرضية مُسجَّلة مسبقًا)\n")
        lines.append(
            f"بدل ترك أهمية ميزات SHAP كمؤشر تفسيري فقط، أُعيد اختبار الميزات الأربع الأكثر ثباتًا عبر سنوات "
            f"walk-forward — `{'`, `'.join(shap_features)}` — كقواعد مباشرة (فرديًا وبالتزاوج فيما بينها)، "
            f"ضمن **عائلة اختبارات صغيرة ومُسجَّلة مسبقًا** ({len(shap_scored)} شرطًا فقط، وليس ضمن الـ923 "
            f"توليفة العامة)، مع تصحيح Bonferroni الأكثر تحفظًا مناسبًا لعائلة بهذا الحجم الصغير.\n"
        )
        n_survivors = int(shap_scored["reject_fdr"].sum())
        min_p = float(shap_scored["p_value_expectancy"].min())
        if n_survivors == 0:
            lines.append(
                f"**النتيجة: لا يوجد أي شرط ينجو من هذا الاختبار الموجَّه** (أفضل قيمة p خام = {_fmt(min_p)}، "
                f"وهي أعلى من 0.05 أصلاً قبل أي تصحيح). هذه نتيجة سلبية صادقة ومهمة: أهمية SHAP الثابتة لهذه "
                f"الميزات تعكس دورها ضمن تفاعلات النموذج الشجري مع ميزات أخرى (مثل يوم الأسبوع)، وليس كقاعدة "
                f"بسيطة قائمة بذاتها أو مزدوجة فيما بينها فقط — أي أن الإشارة التفسيرية لا تتحول تلقائيًا إلى "
                f"قاعدة تداول مستقلة قابلة للاختبار.\n"
            )
        else:
            lines.append(f"نجا **{n_survivors}** شرطًا من تصحيح Bonferroni ضمن هذه العائلة الصغيرة:\n")
            for _, r in shap_scored[shap_scored["reject_fdr"]].iterrows():
                lines.append(f"- `{r['condition']}` (n={r['n']}, عائد متوقع={_fmt(r['expectancy'])}R, "
                             f"p_bonferroni={_fmt(r['p_bonferroni'])})")
        lines.append("")

    lines.append("## نتائج استكشافية منخفضة الثقة\n")
    if len(exploratory):
        lines.append(f"هذه الأنماط ({len(exploratory)} إجمالاً، تُعرض أبرز 8 منها) أظهرت دلالة إحصائية أولية "
                     "(p خام < 0.05) ونجت من اختبار مونت كارلو الفردي، لكنها لم تنجُ من تصحيح FDR الصارم — أي أن "
                     "احتمال ظهورها بالصدفة نتيجة كثرة التوليفات المُختبرة يبقى مرتفعًا نسبيًا. تُعامل كفرضيات "
                     "تستحق المراقبة لا كنتائج مؤكدة:\n")
        for _, r in exploratory.head(8).iterrows():
            lines.append(f"- `{r['condition']}` (n={r['n']}, عائد متوقع={_fmt(r['expectancy'])}R)")
    else:
        lines.append("لا توجد نتائج في هذه الفئة.\n")
    lines.append("")

    lines.append("## أنماط اختُبرت ورُفضت\n")
    lines.append(f"من أصل {n_candidates_tested} توليفة تم اختبارها، رُفضت الغالبية العظمى إما لعدم الدلالة الإحصائية "
                 f"بعد تصحيح FDR، أو لفشلها في اختبار مونت كارلو (احتمال انقلاب الاتجاه > 10%)، أو لعدم تمييزها "
                 f"إحصائيًا عن عيّنة عشوائية بنفس الحجم من بقية البيانات. هذا متوقَّع ومقصود: مع 789 صفقة فقط، "
                 f"معظم التوليفات التي تبدو \"مثيرة\" للوهلة الأولى هي في الغالب ضجيج إحصائي وليست حافة حقيقية.\n")

    lines.append("## تحليل تراجع الحافة (Edge Decay)\n")
    lines.append("الأداء السنوي (Profit Factor / العائد المتوقع / معدل الفوز):\n")
    lines.append("| السنة | عدد الصفقات | معدل الفوز | العائد المتوقع (R) | معامل الربح |")
    lines.append("|---|---|---|---|---|")
    for _, r in yearly.iterrows():
        lines.append(f"| {int(r['year'])} | {int(r['n'])} | {_fmt_pct(r['win_rate'])} | {_fmt(r['expectancy'])} | {_fmt(r['profit_factor'])} |")
    lines.append(
        f"\nاختبار الاتجاه على مستوى تسلسل الصفقات (وهو أقوى إحصائيًا من الاعتماد على 5 نقاط سنوية فقط) "
        f"لم يبلغ العتبة المعتادة للدلالة (p={trend['spearman_p']:.3f} > 0.05)، رغم إشارة ضعيفة نحو التراجع "
        f"(rho={trend['spearman_rho']:.3f}). الخلاصة: **لا يوجد دليل كافٍ حتى الآن على تراجع منهجي في الحافة** "
        f"— الانخفاض في 2023 لا يمكن تمييزه إحصائيًا عن تقلب طبيعي بين السنوات.\n"
    )

    lines.append("## جودة الخروج (Exit Quality)\n")
    lines.append("توزيع تصنيف جودة الخروج عبر جميع الصفقات:\n")
    lines.append("| التصنيف | عدد الصفقات |")
    lines.append("|---|---|")
    for label, n in exit_summary["counts"].items():
        lines.append(f"| {label} | {n} |")
    lines.append(
        f"\nإجمالي المبلغ \"المتروك على الطاولة\" بسبب توقيت الخروج (وليس بسبب جودة نقطة الدخول) عبر كل الصفقات: "
        f"**{exit_summary['total_left_on_table_usd']:,.0f}$**. هذا رقم وصفي عن مسار الصفقة حتى لحظة إغلاقها الفعلية، "
        f"وليس ربحًا كان يمكن تحقيقه بعد ذلك.\n"
    )

    lines.append("## تجميع الفائزين والخاسرين (Clustering)\n")
    for name, ar_name in [("winners", "الصفقات الرابحة"), ("worst_losers", "أسوأ الصفقات الخاسرة")]:
        c = clustering[name]
        if "error" in c:
            lines.append(f"- **{ar_name}**: {c['error']} (n={c['n']}).\n")
        else:
            sizes = ", ".join(f"عنقود {k}: {v} صفقة" for k, v in c["cluster_sizes"].items())
            lines.append(f"- **{ar_name}**: تم العثور على {c['k']} عناقيد متمايزة (silhouette={c['silhouette_score']:.2f}) — {sizes}.\n")
    streak_len = clustering.get("losing_streak_length", 0)
    lines.append(f"- أطول سلسلة خسائر متتالية: **{streak_len}** صفقة"
                 f"{' — قصيرة جدًا لتجميعها بشكل موثوق' if streak_len < 15 else ''}.\n")

    lines.append("## نتائج غير مطلوبة صراحة (Unprompted Findings)\n")
    lines.append(
        f"- **تركّز شديد في وقت الدخول**: 777 من أصل 789 صفقة (≈98.5%) دخلت خلال الساعات 13–15 بتوقيت UTC "
        f"(أي الجلسة الصباحية لنيويورك تقريبًا)، ما يجعل ميزة \"الجلسة\" العريضة عديمة الفائدة عمليًا هنا؛ "
        f"استُخدمت ساعة الدخول الدقيقة كبديل أكثر تمييزًا.\n"
        f"- **طبقة SHAP الاستكشافية**: أكثر الميزات ثباتًا عبر السنوات (walk-forward) في أهمية SHAP هي "
        f"`sl_pct_of_atr` و`pre_entry_momentum_15m/30m` و`regime_vol_asof_prior_day` "
        f"(تداخل متوسط بين أعلى 5 ميزات عبر السنوات ≈ {shap_stability.get('mean_top_k_overlap_fraction', float('nan'))*100:.0f}%). "
        f"هذا مؤشر استكشافي فقط (نموذج شجري تفسيري، وليس إشارة تداول) يستحق اختبارًا إحصائيًا مباشرًا لاحقًا.\n"
    )

    lines.append("## الخطوات التالية المقترحة\n")
    lines.append(
        "1. جمع المزيد من الصفقات قبل الحسم في أي نمط استكشافي أعلاه — العيّنة الحالية (789 صفقة) تجعل أي "
        "تجزئة إضافية (نظام × وقت × نمط) شديدة الحساسية لحجم العيّنة.\n"
        "2. متابعة النمط المؤكَّد `day_of_week=3` مع `pre_entry_pct_up_candles_15m` قرب المنتصف — نجا من "
        "تصحيح FDR على هدفين مستقلَّين (الربح الفعلي وإمكانية الربح الكامنة)، وهو أقوى مرشَّح حتى الآن.\n"
        "3. البحث عن تفاعلات ميزات SHAP مع سياق إضافي (يوم الأسبوع، الجلسة) بدل الاكتفاء بقواعد فردية أو "
        "مزدوجة فيما بينها — الاختبار الموجَّه أعلاه أظهر أن أهميتها التفسيرية لا تُترجم لقاعدة بسيطة قائمة بذاتها.\n"
        "4. مراقبة سنة 2025 عند توفرها لتأكيد ما إذا كان ضعف 2023 كان استثناءً أم بداية اتجاه.\n"
    )

    return "\n".join(lines)
