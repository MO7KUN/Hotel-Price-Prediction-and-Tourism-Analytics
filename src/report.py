"""Create figures, statistics, full documentation, notebook and offline dashboard."""
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from scipy.stats import kruskal,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .prepare import FEATURES,AMENITIES,CLUSTER_FEATURES

FIG=Path('reports/figures')
COLORS={'Economy':'#27877c','Mid-range':'#23405a','Upscale':'#cb7d38'}

def table(frame):
    frame=frame.astype(str)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in frame.values)

def save(name,caption='Booking.com via Silvia & Óscar / Zenodo · quoted stay: 24–25 January 2026'):
    fig=plt.gcf();fig.text(.012,.012,caption,fontsize=8,color='#5a6570')
    fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(FIG/f'{name}.png',dpi=170,facecolor='white')
    fig.savefig(FIG/f'{name}.svg',facecolor='white');plt.close(fig)

def statistics(data):
    # One row per property for descriptive association tests, avoiding repeated-quote pseudo-replication.
    properties=data.groupby('property_id').agg(price=('price_per_night_eur','median'),
        center_km=('distance_center_km','median'),reviews=('review_count','median'),amenity_mentions=('amenity_mentions_count','median'),district=('district','first'))
    correlations={}
    for feature in ['center_km','reviews','amenity_mentions']:
        valid=properties[[feature,'price']].dropna();r=spearmanr(valid[feature],valid.price)
        correlations[feature]={'spearman_rho':float(r.statistic),'p_value':float(r.pvalue),'properties':len(valid)}
    groups=[g.price.to_numpy() for _,g in properties.groupby('district') if len(g)>=3]
    test=kruskal(*groups)
    result={'unit':'one property; median observed quote over available adult configurations',
        'correlations':correlations,'district_kruskal_wallis':{'statistic':float(test.statistic),'p_value':float(test.pvalue),
        'groups_with_at_least_three_properties':len(groups)},'interpretation':'Exploratory associations; unadjusted p-values, selected search sample, no causal identification.'}
    Path('reports/statistical_results.json').write_text(json.dumps(result,indent=2))
    data.groupby('district').agg(quotes=('property_id','size'),properties=('property_id','nunique'),
        median_price_eur=('price_per_night_eur','median'),mean_price_eur=('price_per_night_eur','mean')).sort_values('median_price_eur').to_csv('reports/district_summary.csv')
    return result

def report():
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
        'axes.titlesize':14,'axes.labelsize':10,'figure.figsize':(9,5)})
    data=pd.read_csv('reports/all_quotes_predictions.csv');test=pd.read_csv('reports/test_predictions.csv')
    profiles=pd.read_csv('reports/property_profiles.csv');m=json.loads(Path('reports/metrics.json').read_text())
    audit=json.loads(Path('reports/data_audit.json').read_text());imp=pd.read_csv('reports/importance.csv')
    stats=statistics(data)
    plt.figure();plt.hist(data.price_per_night_eur,bins=24,color=COLORS['Mid-range']);plt.xlabel('Advertised one-room quote (€ / night)');plt.ylabel('Quote count');plt.title('Distribution of observed room quotes');save('price_distribution')
    groups=list(data.groupby('district'))
    plt.figure(figsize=(11,6));plt.boxplot([g.price_per_night_eur for _,g in groups],tick_labels=[n for n,_ in groups],orientation='horizontal',showfliers=True)
    plt.xlabel('Advertised room quote (€ / night)');plt.ylabel('District');plt.title('Location and price · all observed quotes');save('district_prices')
    plt.figure();
    for adults,g in data.groupby('adults'):plt.scatter(g.distance_center_km,g.price_per_night_eur,s=25,alpha=.55,label=f'{int(adults)} adult(s) + infant')
    plt.xlabel('Distance to city centre (km)');plt.ylabel('Advertised room quote (€ / night)');plt.legend();plt.title('Distance, occupation and quoted price');save('distance_prices')
    plt.figure();plt.boxplot([g.price_per_night_eur for _,g in data.groupby('adults')],tick_labels=[str(int(n)) for n,_ in data.groupby('adults')])
    plt.xlabel('Requested adults (plus one infant aged 0)');plt.ylabel('Advertised room quote (€ / night)');plt.title('Occupancy comparisons involve different available hotels');save('occupancy_prices')
    amenity=data.groupby('property_id')[list(AMENITIES)].max().mean().sort_values()
    plt.figure(figsize=(10,6));plt.barh([s.replace('mentions_','').replace('_',' ') for s in amenity.index],amenity.values*100,color=COLORS['Mid-range'])
    plt.xlabel('Properties whose description mentions the service (%)');plt.ylabel('Text keyword');plt.title('Services mentioned in descriptions · presence is not verified availability');save('amenity_mentions')
    compare=pd.DataFrame([{'model':name,'mae':result['mae_eur']} for name,result in m['validation'].items()])
    plt.figure();plt.bar(compare.model,compare.mae,color=COLORS['Mid-range']);plt.ylabel('Validation MAE (€ / night)');plt.xlabel('Model family');plt.title('Estimator comparison · property-disjoint validation')
    save('model_comparison','Booking.com via Zenodo · validation properties only; hyperparameters tuned on training groups')
    plt.figure();plt.scatter(test.price_per_night_eur,test.prediction_eur,color=COLORS['Mid-range'],s=35,alpha=.7)
    upper=max(test.price_per_night_eur.max(),test.prediction_eur.max())*1.05
    plt.plot([0,upper],[0,upper],linestyle='--',color=COLORS['Upscale'],label='Perfect prediction');plt.legend()
    plt.xlabel('Observed quote (€ / night)');plt.ylabel('Predicted quote (€ / night)');plt.title('Final test · hotels excluded from model training');save('predictions')
    plt.figure();plt.scatter(test.prediction_eur,test.prediction_eur-test.price_per_night_eur,s=35,color=COLORS['Mid-range'])
    plt.axhline(0,color=COLORS['Upscale'],linestyle='--');plt.xlabel('Prediction (€ / night)');plt.ylabel('Prediction minus observation (€ / night)');plt.title('Held-out residuals and expensive-offer errors');save('residuals')
    top=imp.head(10).iloc[::-1];plt.figure(figsize=(10,6));plt.barh(top.feature,top.mae_increase_eur,xerr=top.std_eur,color=COLORS['Mid-range'])
    plt.xlabel('Increase in MAE after shuffling (€ / night; mean ± SD)');plt.ylabel('Original input feature');plt.title('Permutation importance · calibration hotels')
    save('importance','Booking.com via Zenodo · calibration quotes, 10 permutations; correlated quote rows share hotels')
    ks=m['clustering']['silhouette_by_k'];plt.figure();plt.plot([int(k) for k in ks],list(ks.values()),marker='o',color=COLORS['Mid-range']);plt.axvline(3,color=COLORS['Upscale'],linestyle='--',label='Requested 3-label taxonomy')
    plt.xlabel('Number of clusters (k)');plt.ylabel('Silhouette score (higher is better)');plt.legend();plt.title('Segmentation diagnostics · three classes retained by design')
    save('silhouette','Booking.com via Zenodo · one profile per training/validation property; no calibration/test centroid fitting')
    plt.figure();
    for label in COLORS:
        g=profiles.loc[profiles.positioning==label];plt.scatter(g.pc1,g.pc2,s=25,alpha=.65,label=f'{label} (n={len(g)})',color=COLORS[label])
    plt.legend();plt.xlabel(f'PC1 ({m["clustering"]["pca_explained_variance"][0]:.0%} historical variance)');plt.ylabel(f'PC2 ({m["clustering"]["pca_explained_variance"][1]:.0%} historical variance)');plt.title('Property positioning · PCA display of four clustering inputs');save('positioning')
    plt.figure();plt.boxplot([profiles.loc[profiles.positioning==label,'median_price_eur'] for label in COLORS],tick_labels=list(COLORS))
    plt.xlabel('Relative positioning label');plt.ylabel('Median observed property quote (€ / night)');plt.title('Positioning uses price as an input · descriptive separation');save('segment_prices')
    Path('examples').mkdir(exist_ok=True)
    test.sort_values(['property_id','adults']).head(1)[['name']+FEATURES].to_csv('examples/hotel.csv',index=False)
    write_readme(m,audit,stats,imp,test,profiles)
    write_dashboard(data,profiles,m)
    write_notebook()
    names=['pandas','numpy','scikit-learn','xgboost','scipy','matplotlib','requests','beautifulsoup4','joblib','pytest']
    versions={name:importlib.metadata.version(name) for name in names}
    Path('requirements-lock.txt').write_text('\n'.join(f'{name}=={version}' for name,version in versions.items())+'\n')
    Path('reports/environment.json').write_text(json.dumps({'python':platform.python_version(),'packages':versions},indent=2))
    print('Generated README, 12 PNG/SVG figure pairs, notebook and interactive tourism dashboard.',flush=True)

def write_readme(m,audit,stats,imp,test,profiles):
    manifest=json.loads(Path('data/source_manifest.json').read_text(encoding='utf-8'))
    live=json.loads(Path('reports/live_collection_attempt.json').read_text()) if Path('reports/live_collection_attempt.json').exists() else {'status':'not executed'}
    score=m['test']['selected']
    results=table(pd.DataFrame([{'Estimator / baseline':name,'Quotes':r['n'],'MAE (€)':f'{r["mae_eur"]:.2f}',
        'RMSE (€)':f'{r["rmse_eur"]:.2f}','R²':f'{r["r2"]:.3f}','Median APE':f'{r["median_ape_percent"]:.1f}%'} for name,r in m['test'].items()]))
    validation=table(pd.DataFrame([{'Family':name,'Grouped CV MAE (€)':f'{m["search"][name]["cv_mae_eur"]:.2f}',
        'Validation MAE (€)':f'{r["mae_eur"]:.2f}','Best hyperparameters':json.dumps(m['search'][name]['best_params'])} for name,r in m['validation'].items()]))
    split=table(pd.DataFrame([{'Role':role,**counts} for role,counts in m['splits'].items()]))
    associations=table(pd.DataFrame([{'Indicator':name,'Property-level Spearman rho':f'{v["spearman_rho"]:.3f}',
        'Unadjusted p':f'{v["p_value"]:.4g}','Properties':v['properties']} for name,v in stats['correlations'].items()]))
    cluster_summary=table(pd.read_csv('reports/cluster_summary.csv').round(2))
    sample=test.sort_values(['property_id','adults']).head(5)[['name','adults','price_per_night_eur','prediction_eur','lower90_eur','upper90_eur']].copy()
    for column in sample.columns[2:]:sample[column]=sample[column].map(lambda x:f'€{x:.2f}')
    example=table(sample)
    baseline=min(m['test']['global_median']['mae_eur'],m['test']['district_occupancy_median']['mae_eur'])
    verdict=('The selected model improves on both tested simple baselines on this held-out sample.' if score['mae_eur']<baseline else 'The selected model does not beat every simple baseline on this held-out sample; the full comparison below is part of the result, not hidden.')
    text=f'''# Hotel Price Prediction and Tourism Analytics

**Analyse des hôtels et prédiction des tarifs touristiques**

A complete, executed educational data-science project using a licensed **Booking.com research snapshot**: automatic collection, data-quality checks, room-price regression, Random Forest / Gradient Boosting / XGBoost tuning, K-Means positioning, statistical analysis and an interactive tourism dashboard.

**Measured results:** {audit['retained_rows']} valid one-room quotes for **{audit['properties']} establishments**, across **{audit['districts']} Barcelona districts**. Selected estimator: **{m['selected_model']}**. Property-disjoint test MAE **€{score['mae_eur']:.2f} / night**, RMSE **€{score['rmse_eur']:.2f}**, R² **{score['r2']:.3f}**. {verdict}

![Held-out observed versus predicted room quotes](reports/figures/predictions.png)

## Contents

- [Quick start](#quick-start)
- [Data provenance and licensing](#data-source-provenance-and-licensing)
- [Web scraping](#web-scraping)
- [Schema and cleaning](#data-schema)
- [Evaluation and regression](#leakage-resistant-evaluation)
- [Measured test results](#measured-test-results)
- [Statistical analysis](#statistical-and-visual-analysis)
- [K-Means positioning](#k-means-hotel-positioning)
- [Tourism dashboard](#dashboard)
- [Prediction examples](#real-prediction-examples)
- [Limitations](#limitations-and-next-steps)

## Quick start

Python 3.12 recommended. Run from the repository root:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
python -m src.predict --input examples/hotel.csv --output predictions.csv
python -m http.server 8000
```

Open **[http://localhost:8000/dashboard/index.html](http://localhost:8000/dashboard/index.html)**. The dashboard is also a self-contained HTML file that opens directly in a browser. It embeds real quote rows and runs without an API key, backend, external chart library or network requests.

To reproduce collection, cleaning, training, figures, documentation and dashboard:

```bash
python -m src.run
```

To train from the committed processed cohort without downloading data:

```bash
python -m src.train
```

To intentionally refresh the original snapshot:

```bash
python -m src.collect --refresh
python -m src.run --skip-download
```

The snapshot is less than 1 MB; model training evaluates 36 cross-validation fits plus refits. Direct package versions used for the executed results are in [requirements-lock.txt](requirements-lock.txt); transitive dependencies are not fully pinned. [Environment details](reports/environment.json) record the execution versions. Fitted artifacts are included and can be regenerated. Load only trusted joblib files.

## Objectives and delivered components

- **Automatic collection:** versioned Booking.com-derived data retrieved programmatically from Zenodo, with source MD5 and local SHA256 verification. An optional direct HTML scraper is also implemented and tested.
- **Price factors:** district, centrality, beach proximity, requested adults, review count, sustainability certification and service mentions extracted from descriptions.
- **Regression:** Random Forest, histogram Gradient Boosting and XGBoost; grouped cross-validation and independent validation/calibration/test hotel sets.
- **Segmentation:** three property groups named **Economy**, **Mid-range** and **Upscale**, ordered by their training-property median quoted price.
- **Dashboard:** linked filters, updated KPIs, district medians, distance/price scatter plot, sortable hotel quotes, held-out predictions and CSV export.

This is a bounded study of **advertised search-result quotes** for **24–25 January 2026**, captured in November 2025. It does not predict realized transaction prices or current hotel rates. All retained offers request one room, one night, 1–3 adults and one child aged **0**. It should not be applied to other dates or occupancy patterns without new data and evaluation.

## Data source, provenance and licensing

Dataset: **Silvia & Óscar (2025), “Alojamientos y precios en Booking.com (Barcelona, 2026)”**, [Zenodo record](https://zenodo.org/records/17560799), DOI [10.5281/zenodo.17560799](https://doi.org/10.5281/zenodo.17560799).

- Published: **{manifest['publication_date']}**; download timestamp, exact URL, byte count and hashes: [source manifest](data/source_manifest.json).
- Data license verified from the Zenodo API: **CC BY-NC-SA 4.0** (attribution, noncommercial use, share alike). Derived data, figures, documentation and fitted research artifacts retain these conditions; source code is MIT licensed separately. See [DATA_LICENSE.md](DATA_LICENSE.md) and [LICENSE](LICENSE).
- The original CSV contains 450 search-result rows, 194 distinct names and 14 source fields in Spanish. The model cohort is a deliberately filtered subset.
- Raw query strings include tracking/session parameters. They are kept only in ignored local raw storage; committed outputs contain canonical property URLs with **all query strings removed**.
- No synthetic rows are added. Currency, guest scores, star categories and seasonal variation are not invented.

The committed processed dataset and fitted artifacts allow local use even when live websites change. The downloader refuses an unexpected source checksum change unless a refresh is explicitly requested.

## Web scraping

`src/collect.py` is the reliable automatic acquisition path. It downloads the authors' published scraped snapshot rather than re-requesting hundreds of hotel pages.

`src/scrape.py` demonstrates direct Booking.com HTML extraction using `requests` and BeautifulSoup. It checks `robots.txt`, spaces requests by at least three seconds, caches successful HTML, strips tracking parameters and rejects empty/block pages. It stops on access failures, restrictions or absent listing cards. It does not bypass a challenge or use proxies.

Example optional live request:

```bash
python -m src.scrape "https://www.booking.com/searchresults.html?ss=Barcelona&checkin=2026-11-14&checkout=2026-11-15&group_adults=2&no_rooms=1&group_children=1&age=0&selected_currency=EUR"
```

**Actual live attempt:** status `{live['status']}`. {live.get('reason','')} The execution record is in [live_collection_attempt.json](reports/live_collection_attempt.json). A response without HTML property cards cannot be treated as successful collection; the executed study uses the published snapshot. New live requests are never silently mixed into the historical cohort.

## Data schema

```mermaid
erDiagram
    SOURCE_QUOTES ||--o{{ CLEAN_QUOTES : validate_and_normalize
    PROPERTIES ||--o{{ CLEAN_QUOTES : property_id
    PROPERTIES ||--|| PROPERTY_SPLITS : property_id
    PROPERTIES ||--|| POSITIONING : property_id
    CLEAN_QUOTES ||--|| PREDICTIONS : quote_row
    SOURCE_QUOTES {{
        string nombre
        string precio
        int adultos
        int habitaciones
        string link
        string descripcion
    }}
    CLEAN_QUOTES {{
        string property_id FK
        date checkin
        date checkout
        int adults
        int rooms
        float price_per_night_eur
        float distance_center_km
    }}
    PROPERTIES {{
        string property_id PK
        string name
        string canonical_url
        string district
    }}
    PROPERTY_SPLITS {{
        string property_id FK
        string split
    }}
    POSITIONING {{
        string property_id FK
        int cluster
        string positioning
    }}
    PREDICTIONS {{
        float prediction_eur
        float lower90_eur
        float upper90_eur
    }}
```

## Cleaning and target definition

1. Normalize Spanish column names and parse explicit EUR price strings (`€ 1.234,50` → 1234.50), review counts (`7.898 comentarios` → 7,898 reviews), and distances (`300 m` → 0.3 km).
2. Extract dates, adults, children, room request and child age from the source URL. Cross-check occupancy against the CSV; reject unexpected adult/child mismatches.
3. Exclude **{audit['excluded_room_count_mismatch']} room-count mismatches**. These rows claim two rooms while their Booking URL requests one, so room count is not safely interpretable.
4. Exclude another **{audit['excluded_multiroom_rows']} matching multi-room search rows**. Requested room count is not sufficient evidence that the displayed price covers all those rooms, so we do not divide the price by that count.
5. Deduplicate canonical property + dates + party configuration. Removed duplicate count: **{audit['duplicate_search_rows_removed']}**. Retain {audit['retained_rows']} one-room rows; target = **quoted price / number of nights**. Every retained stay is exactly one night, so target equals the displayed source price.
6. Create stable property IDs from canonical property paths. Keep names and URLs for display, never as price-model inputs.
7. Extract **text mentions** of nine amenities. A zero means the keyword was absent, not that the amenity does not exist. A description might mention a nearby or paid service; these are imperfect text proxies rather than verified facilities.
8. Keep missing beach distances as missing (not zero); an explicit missingness indicator is added. The observed missing fraction is **{audit['missing_percent']['distance_beach_km']:.1f}%**. Imputation and encoding are fitted only on each training fold.

Price range in the retained cohort: **€{audit['price_summary_eur']['min']:.0f}–€{audit['price_summary_eur']['max']:.0f}**, median **€{audit['price_summary_eur']['50%']:.0f} / night**. No arbitrary price cap, winsorization or test-set-based outlier removal is used. [Data audit](reports/data_audit.json) contains full counts and missingness.

**Inputs ({len(FEATURES)}):** {', '.join(FEATURES)}.

There is **no usable guest-review score or official star category** in this source. The `comentarios` field is a review count, not a rating. City, date, infant count, infant age, room count and stay duration are constants in this cohort and are excluded from learning. Full descriptions, property IDs/names, URLs, observed prices and positioning labels are excluded from regression features.

## Leakage-resistant evaluation

```mermaid
flowchart LR
    A[Versioned Booking research CSV] --> B[Validate room counts and parse units]
    B --> C[225 one-room quotes]
    C --> D[Split by property ID]
    D --> E[Training: grouped 3-fold search]
    E --> F[Independent validation: choose family]
    F --> G[Refit training + validation hotels]
    G --> H[Calibration hotels: residual intervals]
    G --> I[Untouched test hotels: final scores]
    C --> J[One observed profile per property]
    J --> K[Training profiles: scaling + KMeans]
    K --> L[Assign economy / mid-range / upscale]
```

Properties are deterministically shuffled with seed 42 into approximately 60% training, 15% validation, 10% calibration and 15% test groups. All quotes for a property stay together. Consequently, the final price score measures generalization to **unseen establishments**, not memorization of another quote for the same hotel.

{split}

This is a grouped cross-sectional split, **not a temporal forecast**. A single stay date cannot support an honest seasonal backtest. All quality and segmentation choices are documented without using final test scores to choose the price estimator.

## Regression and hyperparameter optimization

Numeric imputation (training median), standardization and categorical mode imputation / one-hot encoding are inside the model pipeline. Unseen districts are encoded without raising an error. A `log1p` price target reduces the influence of the long upper tail. Inversion returns EUR; negative predictions are clipped at zero for evaluation and inference.

Three families each use four grid combinations and 3-fold **GroupKFold by hotel** (12 combinations, 36 CV fits):

- Random Forest: 200 trees, depths 6/unlimited, minimum leaf sizes 2/5.
- Histogram Gradient Boosting: 180 iterations, 7/15 leaf nodes, L2 regularization 1/10, internal random early stopping disabled.
- XGBoost: 200 trees, learning rate 0.05, depth 2/4, lambda 1/10, histogram tree method.

Search scoring is MAE in original EUR despite fitting the log target. Each family's best candidate is evaluated on independent validation hotels. The smallest validation MAE selects **{m['selected_model']}**; it is then refitted on training plus validation hotels. Calibration and test properties remain outside that fit.

{validation}

Full fold and candidate results are in `reports/search_*.csv`.

## Measured test results

{results}

Both baselines use the same final test rows as the estimator. The global median is computed on training plus validation quotes. The district/occupancy baseline uses the median for each training district/adult-count combination and falls back to the global median when absent.

Property-cluster bootstrap (2,000 resamples) gives an approximate 95% test-MAE interval of **€{m['test_mae_property_bootstrap_95ci_eur'][0]:.2f}–€{m['test_mae_property_bootstrap_95ci_eur'][1]:.2f}**. This describes sampling uncertainty on a small selected test set, not a guarantee about future markets. R² can be unstable or negative on small groups; all measured values are kept.

![Model validation comparison](reports/figures/model_comparison.png)
![Test residuals](reports/figures/residuals.png)
![Input importance](reports/figures/importance.png)

The top measured permutation effects are **{', '.join(imp.head(3).feature)}**. These are associations in a fitted predictive model, not causal price effects. Importance uses 10 permutations on calibration quotes; repeated quote rows and correlated features limit its interpretation.

Intervals use each calibration property's **maximum absolute log residual across its observed quotes**. A finite-sample order statistic gives a nominal 90% property-level residual band. Observed test quote coverage is **{m['interval']['test_coverage']:.1%}** and all-quotes-covered test-property coverage is **{m['interval']['test_property_coverage']:.1%}**; median interval width **€{m['interval']['median_width_eur']:.2f}**. There are only {m['interval']['calibration_properties']} calibration properties, so bands are coarse. Exchangeability with future hotel markets is not established and coverage for unobserved party configurations is not claimed.

## Statistical and visual analysis

![Observed quote distribution](reports/figures/price_distribution.png)
![District price distributions](reports/figures/district_prices.png)
![Distance and occupancy](reports/figures/distance_prices.png)
![Occupancy and price](reports/figures/occupancy_prices.png)
![Amenities in descriptions](reports/figures/amenity_mentions.png)

For exploratory inferential statistics, we use **one median-price row per property** rather than treating repeated quotes for the same establishment as independent observations.

{associations}

District Kruskal–Wallis statistic: **{stats['district_kruskal_wallis']['statistic']:.2f}**, unadjusted p **{stats['district_kruskal_wallis']['p_value']:.4g}**, across districts containing at least three properties. This is an exploratory difference in distributions, not a controlled estimate of location's causal effect. The snapshot is a selected top-search-result sample, p-values are unadjusted and occupancy mixes differ between properties. See [statistical results](reports/statistical_results.json) and [district summary](reports/district_summary.csv).

## K-Means hotel positioning

Clustering uses one row per establishment: **{', '.join(CLUSTER_FEATURES)}**. Median price is computed over the available 1–3-adult quotes for that property; not every hotel appears for every occupancy configuration. This can affect positioning and is explicitly a descriptive limitation.

The imputer, scaler and centroids are fitted only on training/validation properties. After standardization, feature weights are 2.0 for log median price, 0.5 for centre distance, 0.25 for log review count and 0.5 for amenity mentions. This explicit price-led positioning rule prevents review popularity or service-text length from dominating the requested economic tiers; it is a business design choice, not an optimized claim about hotel quality. **k=3 is fixed to deliver the requested economy/mid-range/upscale taxonomy.** Scores for k=2–6 are reported as diagnostics; the best diagnostic k is **{m['clustering']['best_diagnostic_k']}**, which is not silently substituted for three. Seed 42 and 30 initializations are used.

Cluster IDs are arbitrary. The three training clusters are named by ascending training-property median observed price. These are **relative market-positioning labels**, not official hotel categories, quality guarantees or supervised classification accuracy claims. K-Means assignment for a new property requires an observed median quote profile; price prediction itself does not require an observed price or a cluster label.

{cluster_summary}

![Silhouette diagnostics](reports/figures/silhouette.png)
![Property segmentation](reports/figures/positioning.png)
![Observed prices by relative segment](reports/figures/segment_prices.png)

PCA is used only to display clusters in two dimensions; K-Means operates on all four weighted standardized inputs. Because price is included in clustering, price separation between the resulting groups is expected and is not independent validation of a pricing model. Group price ranges may overlap. Assigned property labels and cluster inputs are in [property_profiles.csv](reports/property_profiles.csv).

## Dashboard

Open [dashboard/index.html](dashboard/index.html) directly, or serve the repository using `python -m http.server 8000`.

![Interactive tourist analysis dashboard](dashboard/preview.jpg)

- Filter by district, adults, positioning and property name; reset filters with one click.
- Inspect dynamically updated quote counts, property counts, median price and central distance.
- Compare district medians and the central-distance/price relationship with units and source context.
- Sort quotes and open canonical source property links; export the filtered real rows to CSV.
- Inspect **test-only** observed/predicted prices separately from the full descriptive dataset. Training estimates are not presented as holdout results.
- The dashboard's model-quality panel reports the fixed held-out test score, independently of exploratory filters.

The dashboard displays the dated snapshot, not live availability. Only a partial search sample is observed; it is not a complete Barcelona hotel inventory. Data and code use no external browser dependencies, so the dashboard can be opened offline.

## Real prediction examples

These are actual held-out properties and quotes, not illustrative synthetic hotel records:

{example}

```bash
python -m src.predict --input examples/hotel.csv --output examples/prediction.csv
```

All {len(FEATURES)} named input columns are required. Numeric missingness is imputed inside the fitted pipeline. `adults` must remain in the observed 1–3 range; binary service-mention columns accept 0/1. Prices returned are EUR per night for the fixed one-room, one-infant, one-night research context. Other dates, cities, children or room types require new collection and retraining; the interface does not manufacture seasonal effects.

## Reproducibility and project structure

```text
src/collect.py     Versioned Zenodo download and checksums
src/scrape.py      Optional respectful Booking HTML scraper
src/prepare.py     Occupancy checks, unit parsing and feature extraction
src/model.py       Shared preprocessing, metrics and input validation
src/train.py       Property splits, three grid searches, intervals and KMeans
src/predict.py     Batch room-price inference and observed-profile assignment
src/report.py      Statistics, 12 figures, README, notebook and dashboard
src/run.py         End-to-end entry point
artifacts/        Fitted regression and positioning models
data/processed/   Canonical sanitized quote dataset
dashboard/        Self-contained interactive tourism analysis
reports/          Measured scores, splits, profiles, statistics and PNG/SVG figures
notebooks/        Exploratory analysis notebook
examples/         Real held-out input and model output
tests/            Parsing, occupancy, leakage and inference contracts
.github/          Automated tests on push / pull request
```

The notebook requires Jupyter separately. The pipeline and dashboard do not. Graphs are exported as both PNG for README rendering and SVG for reuse. The [verification record](reports/verification.json) records the actual local checks; GitHub Actions provides separate remote test status.

## Limitations and next steps

- **One city and one stay date:** seasonal effects, multi-city transferability and temporal drift cannot be estimated. A future collection should repeat the same hotels across seasons with fixed occupancy, room type and booking lead time, then use a temporal backtest.
- **Missing category/rating fields:** neither official stars nor guest scores are available. We retain review counts and text-derived service mentions with their proper meanings.
- **Offer comparability:** actual room type, cancellation conditions, meals and tax inclusion are not recorded. Advertised search-result prices can represent different room offers, even under the same requested occupancy.
- **Scraper availability:** the live request did not yield parseable listing cards. This project does not claim successful fresh Booking collection; the automatic published-snapshot route is the executed source.
- **Small selected sample:** only the first 75 results per search configuration were published. Filtering leaves {audit['retained_rows']} quotes and {audit['properties']} properties; validation and test estimates have substantial uncertainty.
- **Keyword noise:** amenities may be mentioned negatively, as nearby facilities or as paid extras. These inputs measure text mentions, not confirmed inclusion in a tariff.
- **Observed positioning:** clusters use median observed prices and an uneven occupancy mix; labels are relative to this sample and may be unstable under another collection.
- **No causal inference:** correlations, group tests and permutation importance reveal associations. They cannot show that adding a service would raise a hotel's price by a given amount.
- **Research use:** the data license is noncommercial and share alike. A commercial pricing system needs appropriately licensed data and substantially broader validation.

## References

- [Original Booking.com dataset, authors, version and license](https://zenodo.org/records/17560799)
- [CC BY-NC-SA 4.0 terms](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- [scikit-learn grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
- [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [scikit-learn K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)

No affiliation with Booking.com, Hotels.com or the dataset authors is implied.
'''
    Path('README.md').write_text(text,encoding='utf-8')

def write_dashboard(data,profiles,m):
    data=data.merge(profiles[['property_id','positioning']],on='property_id',validate='many_to_one')
    keep=['property_id','name','canonical_url','district','adults','distance_center_km','price_per_night_eur','positioning','split','prediction_eur']
    payload=json.dumps(data[keep].to_dict('records'),ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    metrics_payload=json.dumps({'mae':m['test']['selected']['mae_eur'],'r2':m['test']['selected']['r2'],
        'model':m['selected_model'],'test_quotes':m['test']['selected']['n'],'test_properties':m['splits']['test']['properties']})
    template=Path('dashboard/template.html').read_text(encoding='utf-8')
    Path('dashboard/index.html').write_text(template.replace('__DATA_PAYLOAD__',payload).replace('__MODEL_PAYLOAD__',metrics_payload),encoding='utf-8')

def write_notebook():
    cells=[]
    def md(text):cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
    def code(text):cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.splitlines(keepends=True)})
    md('# Analyse des hôtels et prédiction des tarifs touristiques\nBooking.com via Silvia & Óscar / Zenodo. Research snapshot published November 2025 for 24–25 January 2026. Source data and derivative outputs: CC BY-NC-SA 4.0.')
    code("from pathlib import Path\nimport os, json\nif Path.cwd().name == 'notebooks': os.chdir('..')\nimport pandas as pd\nfrom IPython.display import display, Image\nquotes=pd.read_csv('data/processed/hotel_quotes.csv')\nmetrics=json.loads(Path('reports/metrics.json').read_text())\ndisplay(quotes.head())\ndisplay(pd.Series(json.loads(Path('reports/data_audit.json').read_text())))")
    md('## Préparation et qualité\n75 room-count contradictions and 150 multi-room offers are excluded. Remaining quotes concern one room, one night and one infant. Review counts are not review scores; amenities are description mentions.')
    code("assert quotes.rooms.eq(1).all()\nassert quotes.nights.eq(1).all()\nassert not quotes.canonical_url.str.contains('?',regex=False).any()\ndisplay(Image('reports/figures/district_prices.png'))\ndisplay(Image('reports/figures/amenity_mentions.png'))")
    md('## Évaluation par établissement\nAll observations for one property stay in one partition. Grid search uses GroupKFold; calibration/test hotels do not fit the selected estimator.')
    code("splits=pd.read_csv('reports/property_splits.csv')\nassert splits.groupby('property_id').split.nunique().max()==1\ndisplay(pd.DataFrame(metrics['validation']).T)\ndisplay(pd.DataFrame(metrics['test']).T)\ndisplay(Image('reports/figures/predictions.png'))")
    md('## Positionnement\nThree KMeans groups are ordered by training-cluster median quoted price. They are not official star ratings; observed price is one of the clustering inputs.')
    code("display(pd.read_csv('reports/cluster_summary.csv'))\ndisplay(Image('reports/figures/positioning.png'))")
    md('## Statistiques et limites\nOne aggregated row per property is used for exploratory association tests. One stay date cannot identify seasonality; the test sample is small and selected.')
    code("display(json.loads(Path('reports/statistical_results.json').read_text()))\ndisplay(pd.read_csv('reports/importance.csv'))")
    md('## Exemple réel et tableau de bord\nThe fitted model uses the recorded package versions. Open `dashboard/index.html` in a browser for interactive exploration.')
    code("from src.predict import predict\ndisplay(predict(pd.read_csv('examples/hotel.csv')))")
    Path('notebooks').mkdir(exist_ok=True)
    Path('notebooks/01_hotel_analysis.ipynb').write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5},indent=2),encoding='utf-8')

if __name__=='__main__':report()
