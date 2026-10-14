

# --------------------------------------------------------------------------- #
# 8. fitur discovery (harga relatif alternatif, kebiasaan belanja, peluruhan DPD)
# --------------------------------------------------------------------------- #
def discovery_features(apps, ecom, repay, bureau):
    row = pd.Series(np.arange(len(apps)), index=apps.customer_id.to_numpy())
    INC = apps.set_index('customer_id')['declared_monthly_income']
    out = {}

    # ---------------- e-commerce: alternatif definisi harga relatif ----------------
    items = ecom[ecom.order_status == 'completed'].merge(
        apps[['customer_id', 'decision_datetime']], on='customer_id', how='inner')
    items = items[items.transaction_datetime <= items.decision_datetime].copy()
    items['days_ago'] = (items.decision_datetime - items.transaction_datetime).dt.total_seconds()/86400
    items['unit'] = items.item_total / items.quantity.replace(0, np.nan)
    cat_med = items.unit.groupby(items.product_category).transform('median')
    cat_mean = items.unit.groupby(items.product_category).transform('mean')
    items['lrel_med'] = np.log(items.unit / cat_med)
    items['lrel_mean'] = np.log(items.unit / cat_mean)
    items['above_med'] = (items.unit > cat_med).astype(float)

    # (1) ratio-of-means, bukan mean-of-ratios
    unit_by_cat_mean = items.groupby(['customer_id', 'product_category']).unit.mean().groupby('customer_id').mean()
    out['px_unit_mean'] = items.groupby('customer_id').unit.mean().reindex(apps.customer_id).to_numpy()
    out['px_lrel_mean_of_logs'] = items.groupby('customer_id').lrel_med.mean().reindex(apps.customer_id).to_numpy()
    out['px_lrel_mean_catmean'] = items.groupby('customer_id').lrel_mean.mean().reindex(apps.customer_id).to_numpy()
    out['px_frac_above_med'] = items.groupby('customer_id').above_med.mean().reindex(apps.customer_id).to_numpy()
    out['px_unit_mean_vs_income'] = (out['px_unit_mean'] / INC.reindex(apps.customer_id).to_numpy())


    # (2) window pendek dari harga relatif
    for lo, hi in [(0, 14), (0, 30), (14, 60), (30, 90), (90, 180)]:
        m = (items.days_ago > lo) & (items.days_ago <= hi)
        out[f'px_lrel_{lo}_{hi}'] = items[m].groupby('customer_id').lrel_med.mean().reindex(apps.customer_id).to_numpy()

    # (3) frekuensi & kebiasaan belanja
    items['_day'] = items.transaction_datetime.dt.normalize()
    active_days = items.groupby('customer_id')['_day'].nunique()
    out['ec_active_days'] = active_days.reindex(apps.customer_id).to_numpy()
    out['ec_spend_per_active_day'] = (items.groupby('customer_id').item_total.sum().reindex(apps.customer_id).to_numpy()
                                      / active_days.reindex(apps.customer_id).replace(0, np.nan).to_numpy())
    out['ec_distinct_products'] = items.groupby('customer_id').product_category.nunique().reindex(apps.customer_id).to_numpy()
    out['ec_orders_per_month'] = (items.groupby('customer_id').order_id.nunique().reindex(apps.customer_id).to_numpy()
                                  / (items.groupby('customer_id').days_ago.max().reindex(apps.customer_id).to_numpy()/30.44 + 1))
    night = (items.transaction_datetime.dt.hour >= 22) | (items.transaction_datetime.dt.hour < 5)
    out['ec_night_share'] = night.groupby(items.customer_id).mean().reindex(apps.customer_id).to_numpy()
    out['ec_weekend_share'] = (items.transaction_datetime.dt.dayofweek >= 5).groupby(items.customer_id).mean().reindex(apps.customer_id).to_numpy()
    months = items.assign(m=items.days_ago//30).groupby(['customer_id','m']).item_total.sum()
    out['ec_months_with_spend'] = months.groupby('customer_id').count().reindex(apps.customer_id).to_numpy()
    out['ec_month_max_share'] = (months.groupby('customer_id').max().reindex(apps.customer_id).to_numpy()
                                 / months.groupby('customer_id').sum().reindex(apps.customer_id).replace(0, np.nan).to_numpy())
    out['ec_month_zero_share'] = 1 - out['ec_months_with_spend'] / (items.groupby('customer_id').days_ago.max().reindex(apps.customer_id).to_numpy()/30.44 + 1)

    # ---------------- repayment: peluruhan waktu & durasi tunggakan ----------------
    cut = repay.customer_id.map(apps.set_index('customer_id')['decision_datetime'])
    rh = repay[repay.due_date <= cut].copy()
    rh['days_ago'] = (cut.reindex(rh.index) - rh.due_date).dt.total_seconds()/86400
    for hl in [60, 180]:
        w = np.exp(-rh.days_ago*np.log(2)/hl)
        out[f'rh_decay_dpd{hl}'] = (w*rh.max_days_past_due).groupby(rh.customer_id).sum().reindex(apps.customer_id).to_numpy()
        out[f'rh_decay_late{hl}'] = (w*(rh.max_days_past_due>0)).groupby(rh.customer_id).sum().reindex(apps.customer_id).to_numpy()
    settled = rh.amount_paid >= rh.amount_due
    rh['unpaid'] = np.where(settled, 0.0, rh.amount_due.astype(float))
    lo = rh.groupby(['customer_id','loan_id']).unpaid.sum()
    out['rh_loans_ever_unpaid'] = (lo>0).groupby(level=0).sum().reindex(apps.customer_id).to_numpy()
    out['rh_unpaid_share_loans'] = (lo>0).groupby(level=0).mean().reindex(apps.customer_id).to_numpy()

    out['rh_inst_per_loan'] = rh.groupby(['customer_id','loan_id']).size().groupby(level=0).mean().reindex(apps.customer_id).to_numpy()
    out['rh_paid_ratio_raw'] = (rh.amount_paid/rh.amount_due.replace(0,np.nan)).groupby(rh.customer_id).mean().reindex(apps.customer_id).to_numpy()


    # ---------------- bureau: usia sistem kredit & konsistensi ----------------
    bu = bureau.merge(apps[['customer_id','decision_datetime']], on='customer_id', how='inner')
    bu = bu[bu.available_datetime <= bu.decision_datetime]
    g = bu.groupby('customer_id')
    out['bu_providers'] = bu.groupby('customer_id').bureau_provider.nunique().reindex(apps.customer_id).to_numpy()
    out['bu_reports'] = g.size().reindex(apps.customer_id).to_numpy()



    def _slope(d):
        dd = d[['age_days', 'score']].dropna()
        if len(dd) < 3 or dd.age_days.nunique() < 3:
            return np.nan
        return float(np.polyfit(dd.age_days.to_numpy(), dd.score.to_numpy(), 1)[0])

    bu['age_days'] = (bu.decision_datetime - bu.report_datetime).dt.total_seconds() / 86400
    out['bu_score_slope'] = bu.groupby('customer_id').apply(_slope, include_groups=False).reindex(apps.customer_id).to_numpy()
    last3 = bu.sort_values('available_datetime').groupby('customer_id').tail(3)
    out['bu_score_last3_std'] = last3.groupby('customer_id').score.std().reindex(apps.customer_id).to_numpy()
    out['bu_score_last3_mean'] = last3.groupby('customer_id').score.mean().reindex(apps.customer_id).to_numpy()


    X_new = pd.DataFrame(out, index=range(len(apps)))
    return X_new.replace([np.inf, -np.inf], np.nan)
