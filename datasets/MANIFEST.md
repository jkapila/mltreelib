# Datasets — TabArena-v0.1 (OpenML suite 457, alias `tabarena-v0.1`)

All 51 curated TabArena-v0.1 datasets, downloaded as Parquet from OpenML's
direct dataset mirrors (`parquet_url` in each OpenML data description). Files contain the raw
feature columns plus the target column; categorical features are stored as
dictionary/categorical strings (pandas `Categorical` on read). Any pandas/pyarrow read works:
`pd.read_parquet('datasets/<name>.parquet')`.

Validation (2026-10-28): every file opens with pandas+pyarrow, has non-zero rows, no duplicate
columns, and row counts match TabArena's curated metadata exactly. No failures.

| Dataset | OpenML ID | Task | Target | Rows x Cols | File size | License | Source URL |
|---|---|---|---|---|---|---|---|
| airfoil_self_noise | 46904 | regression | `scaled-sound-pressure` | 1,503 x 6 | 20 KB | CC BY 4.0 | https://www.openml.org/d/46904 |
| Amazon_employee_access | 46905 | classification (binary) | `ResourceApproved` | 32,769 x 10 | 438 KB | Public Domain | https://www.openml.org/d/46905 |
| anneal | 46906 | classification (multiclass) | `classes` | 898 x 39 | 36 KB | CC BY 4.0 | https://www.openml.org/d/46906 |
| Another-Dataset-on-used-Fiat-500 | 46907 | regression | `price` | 1,538 x 8 | 27 KB | CC0: Public Domain | https://www.openml.org/d/46907 |
| APSFailure | 46908 | classification (binary) | `AirPressureSystemFailure` | 76,000 x 171 | 35.1 MB | CC BY 4.0 | https://www.openml.org/d/46908 |
| bank-marketing | 46910 | classification (binary) | `SubscribeTermDeposit` | 45,211 x 14 | 320 KB | CC BY 4.0 | https://www.openml.org/d/46910 |
| Bank_Customer_Churn | 46911 | classification (binary) | `churn` | 10,000 x 11 | 175 KB | Public | https://www.openml.org/d/46911 |
| Bioresponse | 46912 | classification (binary) | `MoleculeElicitsResponse` | 3,751 x 1777 | 3.8 MB | Public Domain | https://www.openml.org/d/46912 |
| blood-transfusion-service-center | 46913 | classification (binary) | `DonatedBloodInMarch2007` | 748 x 5 | 8 KB | CC BY 4.0 | https://www.openml.org/d/46913 |
| churn | 46915 | classification (binary) | `CustomerChurned` | 5,000 x 20 | 150 KB | MIT License | https://www.openml.org/d/46915 |
| coil2000_insurance_policies | 46916 | classification (binary) | `MobileHomePolicy` | 9,822 x 86 | 305 KB | CC BY 4.0 | https://www.openml.org/d/46916 |
| concrete_compressive_strength | 46917 | regression | `ConcreteCompressiveStrength` | 1,030 x 9 | 31 KB | CC BY 4.0 | https://www.openml.org/d/46917 |
| credit-g | 46918 | classification (binary) | `good_or_bad_customer` | 1,000 x 21 | 30 KB | CC BY 4.0 | https://www.openml.org/d/46918 |
| credit_card_clients_default | 46919 | classification (binary) | `DefaultOnPaymentNextMonth` | 30,000 x 24 | 1.6 MB | CC BY 4.0 | https://www.openml.org/d/46919 |
| customer_satisfaction_in_airline | 46920 | classification (binary) | `satisfaction` | 129,880 x 22 | 1.4 MB | CC0: Public Domain | https://www.openml.org/d/46920 |
| diabetes | 46921 | classification (binary) | `TestedPositiveForDiabetes` | 768 x 9 | 18 KB | CC0: Public Domain | https://www.openml.org/d/46921 |
| Diabetes130US | 46922 | classification (binary) | `EarlyReadmission` | 71,518 x 48 | 1000 KB | CC BY 4.0 | https://www.openml.org/d/46922 |
| diamonds | 46923 | regression | `price` | 53,940 x 10 | 576 KB | MIT License | https://www.openml.org/d/46923 |
| E-CommereShippingData | 46924 | classification (binary) | `ArrivedLate` | 10,999 x 11 | 89 KB | Public Domain | https://www.openml.org/d/46924 |
| Fitness_Club | 46927 | classification (binary) | `attended` | 1,500 x 7 | 18 KB | Public Domain | https://www.openml.org/d/46927 |
| Food_Delivery_Time | 46928 | regression | `Time_taken(min)` | 45,451 x 10 | 534 KB | Database Contents License (DbCL) v1.0 | https://www.openml.org/d/46928 |
| GiveMeSomeCredit | 46929 | classification (binary) | `FinancialDistressNextTwoYears` | 150,000 x 11 | 3.3 MB | Public | https://www.openml.org/d/46929 |
| hazelnut-spread-contaminant-detection | 46930 | classification (binary) | `Contaminated` | 2,400 x 31 | 402 KB | CC BY-SA | https://www.openml.org/d/46930 |
| healthcare_insurance_expenses | 46931 | regression | `charges` | 1,338 x 7 | 25 KB | Database Contents License (DbCL) v1.0 | https://www.openml.org/d/46931 |
| heloc | 46932 | classification (binary) | `RiskPerformance` | 10,459 x 24 | 210 KB | Public | https://www.openml.org/d/46932 |
| hiva_agnostic | 46933 | classification (multiclass) | `CompoundActivity` | 3,845 x 1618 | 2.1 MB | Public | https://www.openml.org/d/46933 |
| houses | 46934 | regression | `LnMedianHouseValue` | 20,640 x 9 | 431 KB | Public | https://www.openml.org/d/46934 |
| HR_Analytics_Job_Change_of_Data_Scientists | 46935 | classification (binary) | `LookingForJobChange` | 19,158 x 13 | 132 KB | CC0: Public Domain | https://www.openml.org/d/46935 |
| in_vehicle_coupon_recommendation | 46937 | classification (binary) | `AcceptCoupon` | 12,684 x 25 | 110 KB | CC BY 4.0 | https://www.openml.org/d/46937 |
| Is-this-a-good-customer | 46938 | classification (binary) | `bad_client_target` | 1,723 x 14 | 24 KB | CC0: Public Domain | https://www.openml.org/d/46938 |
| jm1 | 46979 | classification (binary) | `defects` | 10,885 x 22 | 407 KB | Public Domain | https://www.openml.org/d/46979 |
| kddcup09_appetency | 46939 | classification (binary) | `appetency` | 50,000 x 213 | 6.8 MB | Public | https://www.openml.org/d/46939 |
| Marketing_Campaign | 46940 | classification (binary) | `Response` | 2,240 x 26 | 75 KB | Public | https://www.openml.org/d/46940 |
| maternal_health_risk | 46941 | classification (multiclass) | `RiskLevel` | 1,014 x 7 | 10 KB | CC BY 4.0 | https://www.openml.org/d/46941 |
| miami_housing | 46942 | regression | `SALE_PRC` | 13,776 x 16 | 1.0 MB | CC BY-NC-SA | https://www.openml.org/d/46942 |
| MIC | 46980 | classification (multiclass) | `LET_IS` | 1,699 x 112 | 131 KB | CC BY 4.0 | https://www.openml.org/d/46980 |
| NATICUSdroid | 46969 | classification (binary) | `Malware` | 7,491 x 87 | 151 KB | CC BY 4.0 | https://www.openml.org/d/46969 |
| online_shoppers_intention | 46947 | classification (binary) | `Revenue` | 12,330 x 18 | 328 KB | CC BY 4.0 | https://www.openml.org/d/46947 |
| physiochemical_protein | 46949 | regression | `ResidualSize` | 45,730 x 10 | 2.7 MB | CC BY 4.0 | https://www.openml.org/d/46949 |
| polish_companies_bankruptcy | 46950 | classification (binary) | `company_bankrupt` | 5,910 x 65 | 3.1 MB | CC BY 4.0 | https://www.openml.org/d/46950 |
| qsar-biodeg | 46952 | classification (binary) | `Biodegradable` | 1,054 x 42 | 106 KB | CC BY 4.0 | https://www.openml.org/d/46952 |
| QSAR-TID-11 | 46953 | regression | `MEDIAN_PXC50` | 5,742 x 1025 | 1.2 MB | Public | https://www.openml.org/d/46953 |
| QSAR_fish_toxicity | 46954 | regression | `LC50` | 907 x 7 | 25 KB | CC BY 4.0 | https://www.openml.org/d/46954 |
| SDSS17 | 46955 | classification (multiclass) | `ObjectType` | 78,053 x 12 | 6.4 MB | Public Domain | https://www.openml.org/d/46955 |
| seismic-bumps | 46956 | classification (binary) | `HighEnergySeismicBump` | 2,584 x 16 | 54 KB | CC BY 4.0 | https://www.openml.org/d/46956 |
| splice | 46958 | classification (multiclass) | `SiteType` | 3,190 x 61 | 113 KB | CC BY 4.0 | https://www.openml.org/d/46958 |
| students_dropout_and_academic_success | 46960 | classification (multiclass) | `AcademicOutcome` | 4,424 x 37 | 128 KB | CC BY 4.0 | https://www.openml.org/d/46960 |
| superconductivity | 46961 | regression | `critical_temp` | 21,263 x 82 | 7.4 MB | CC BY 4.0 | https://www.openml.org/d/46961 |
| taiwanese_bankruptcy_prediction | 46962 | classification (binary) | `Bankrupt` | 6,819 x 95 | 4.7 MB | CC BY 4.0 | https://www.openml.org/d/46962 |
| website_phishing | 46963 | classification (multiclass) | `WebsiteType` | 1,353 x 10 | 12 KB | CC BY 4.0 | https://www.openml.org/d/46963 |
| wine_quality | 46964 | regression | `median_wine_quality` | 6,497 x 13 | 100 KB | CC BY 4.0 | https://www.openml.org/d/46964 |

**Total: 51 datasets, 87.4 MB** (small <1k rows: 4, medium 1k-50k: 40, large >=50k: 7; 7 regression, 44 classification).

Notes:
- Sizes are on-disk Parquet; the same data as gzip-ARFF from OpenML is larger on disk.
- High-cardinality categorical sets are included, e.g. `Amazon_employee_access` (100% categorical),
  `NATICUSdroid`, `splice`, `hiva_agnostic` (100% categorical), `website_phishing`, `anneal`, `MIC`,
  `Diabetes130US` (83% categorical), `in_vehicle_coupon_recommendation` (88%).
- BeyondArena (TabArena's beyond-IID companion: 103 IID / 21 temporal / 18 grouped tasks over 142
  datasets) is **not** distributed via OpenML; it ships as per-dataset containers through the
  optional `data-foundry` Python package (`data_foundry.collections.BEYOND_ARENA`). It re-curates
  some of the same raw source datasets (e.g. `airfoil_self_noise`) but with temporal/grouped splits,
  so its data is *not* the same files as this collection. Not downloaded here (no plain
  csv/parquet direct URLs; requires the data-foundry toolchain).
- OpenML suite reference: https://www.openml.org/s/457 (alias `tabarena-v0.1`); benchmark home:
  https://github.com/autogluon/tabarena. Licenses are those declared on the OpenML data
  descriptions / TabArena curated metadata.
