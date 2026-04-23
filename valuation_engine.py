import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings("ignore")

# --- BLOCK 1: THE SANITIZER (Phases 3, 4, 5, 7, 15, 16) ---
def industrial_sanitizer(df, target, phase=3, schema_map=None):
    """
    INDUSTRIAL HARDENING ENGINE
    Handles: Universal Schema Mapping, Structural Cleanup, and Outlier Shielding.
    """
    df = df.copy()
    
    # 0. UNIVERSAL SCHEMA BRIDGE & GROUP DESTRUCTION
    if schema_map:
        for csv_col, engine_slot in schema_map.items():
            # Handle both strings and tuples/lists for grouped dropping
            cols_to_process = [csv_col] if isinstance(csv_col, str) else csv_col
            
            if engine_slot in ['PropertyID', 'Full_Name', 'Serial_Number']:
                df = df.drop(columns=[c for c in cols_to_process if c in df.columns], errors='ignore')
            else:
                if isinstance(csv_col, str):
                    df = df.rename(columns={csv_col: engine_slot})

    # 1. STRUCTURAL CLEANUP (Phase 3, 4, 5)
    if phase in [3, 4, 5]:
        internal_rem = ['Serial_Number', 'Full_Name', 'id', 'PropertyID', 'Unnamed: 0']
        df = df.drop(columns=[c for c in internal_rem if c in df.columns], errors='ignore')
        df = df.drop_duplicates().reset_index(drop=True)
        df = df.loc[df.isnull().mean(axis=1) < 0.5].reset_index(drop=True)
        print(f"✅ Phase {phase}: Structural Integrity Secured. Shape: {df.shape}")

    # 2. TYPE STANDARDIZATION & SYNC (Phase 7)
    if phase == 7: 
        # 1. Standardize Features
        for c in df.columns:
            if c != target:
                if df[c].dtype == 'O': 
                    try: 
                        df[c] = pd.to_numeric(df[c])
                    except: 
                        df[c] = df[c].astype(str)
        
        # 2. Standardize Target (Detecting Numeric vs Categorical)
        try:
            df[target] = pd.to_numeric(df[target])
            target_status = "Numeric (Float64)"
        except:
            df[target] = df[target].astype(str).str.strip()
            target_status = "Categorical (String/Label)"

        # 3. Final Integrity Check
        df = df.dropna(subset=[target]).reset_index(drop=True)
        print(f"✅ Final Check: Features synchronized to Numeric. Target identified as {target_status}.")
        print(f"💎 System Status: Industrial Model-Ready for {target_status} tasks.")

    # 3. RESILIENCE HARDENING (Phase 15, 16: Outlier Shield)
    if phase in [15, 16]:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col != target:
                limit = df[col].quantile(0.99)
                df[col] = df[col].clip(upper=limit)
        
        # Prevent Target Shock (Only if target is numeric)
        if df[target].dtype.kind in 'if':
            t_limit = df[target].quantile(0.99)
            df = df[df[target] <= t_limit].reset_index(drop=True)
        
        print(f"✅ Phase {phase}: Outlier Shield Active (99th Percentile Clipping).")

    return df

# --- BLOCK 2: THE MINER (Phases 6, 8, 9, 11) ---

def signal_mining_engine(df, target, phase=6, top_k=10):
    # Ensure we are working with numeric data for mining
    num_df = df.select_dtypes(include=[np.number])
    ncs = [c for c in num_df.columns if c != target]

    # --- PHASE 6: CORRELATION DISCOVERY ---
    if phase == 6:
        return num_df.corr(), df

    # --- PHASE 8: AUTONOMOUS ATOMIC RATIO DISCOVERY ---
    if phase == 8: 
        res = []
        # A. STATISTICAL CORRELATION MINING
        for i, c1 in enumerate(ncs):
            for j, c2 in enumerate(ncs):
                if i != j:
                    score = abs((df[c1] / (df[c2] + 1)).corr(df[target]))
                    res.append((c1, c2, score))
        
        sdf = pd.DataFrame(res, columns=['N', 'D', 'S']).sort_values('S', ascending=False)
        best_pairs = list(zip(sdf['N'], sdf['D']))[:top_k]
        
        # B. 🤖 THE ARCHEТYPE HUNTER (Autonomous Strategic Injection)
        # Targeted search for "Efficiency" and "Depreciation" signals
        time_cols = [c for c in ncs if any(w in c.lower() for w in ['age', 'yrbuilt', 'tenure'])]
        mkt_cols  = [c for c in ncs if any(w in c.lower() for w in ['zhvi', 'px', 'price_index'])]
        qual_cols = [c for c in ncs if any(w in c.lower() for w in ['grade', 'condition', 'quality'])]
        size_cols = [c for c in ncs if any(w in c.lower() for w in ['sqft', 'lot', 'area'])]
        
        archetypes = []
        # Depreciation Logic: Time / Market Strength
        for t in time_cols:
            for m in mkt_cols: archetypes.append((t, m))
        # Quality Density Logic: Quality / Size
        for q in qual_cols:
            for s in size_cols: archetypes.append((q, s))
        # Market Efficiency Logic: Size / Market Strength
        for s in size_cols:
            for m in mkt_cols: archetypes.append((s, m))
            
        # Merge AI discoveries with Strategic Archetypes (Unique set)
        final_pairs = list(dict.fromkeys(best_pairs + archetypes))
        return final_pairs, sdf 

    # --- PHASE 9: POLYNOMIAL DISCOVERY ---
    if phase == 9:
        res = [(c, abs((df[c]**2).corr(df[target]))) for c in ncs]
        sdf = pd.DataFrame(res, columns=['Col', 'S']).sort_values('S', ascending=False)
        best_cols = sdf['Col'].head(top_k).tolist()
        return best_cols, sdf 

    # --- PHASE 11: DEEP FEATURE SYNTHESIS (DFS) ---
    if phase == 11:
        import featuretools as ft
        import logging
        logging.getLogger('featuretools').setLevel(logging.ERROR)
        
        MAX_INPUTS = 200
        # A. Quick Pre-Filter
        X_t = df.drop(columns=[target], errors='ignore').select_dtypes(include=[np.number]).fillna(-999)
        y_t = df[target]
        
        from sklearn.ensemble import RandomForestRegressor
        filter_model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        filter_model.fit(X_t, y_t)
        
        # B. Select Base Columns (Removing previous engineered noise to prevent circular logic)
        importance = pd.Series(filter_model.feature_importances_, index=X_t.columns)
        top_blocks = importance.nlargest(MAX_INPUTS).index.tolist()
        base_cols = [c for c in top_blocks if "_squared" not in c and "_per_" not in c and "_logic" not in c]
        
        # C. Mining Engine (Depth 3)
        es = ft.EntitySet(id="automated_mining")
        df_ft = df[base_cols].reset_index()
        es = es.add_dataframe(dataframe_name="main_table", dataframe=df_ft, index="index")

        feature_matrix, _ = ft.dfs(
            entityset=es,
            target_dataframe_name="main_table",
            trans_primitives=['add_numeric', 'multiply_numeric', 'subtract_numeric', 'divide_numeric'],
            max_depth=2,
            verbose=False
        )

        # D. Clean and Re-Merge
        df_new = feature_matrix.drop(columns=['index'], errors='ignore')
        # Ensure we don't duplicate columns that already existed
        cols_to_add = df_new.columns.difference(df.columns)
        df = pd.concat([df, df_new[cols_to_add]], axis=1)
        
        print(f"✅ Phase 11: DFS Complete. Depth: 3. Features Secured: {len(df.columns)}")
        return None, df

    return [], df

# --- BLOCK 3: THE ARCHITECT (Phases 10, 12, 13, 17) ---

def industrial_feature_architect(df, ratio_pairs=[], poly_cols=[], date_cols=[], age_map={}, target='SalePrice'):
    df = df.copy()
    audit_report = {
        'counts': {}, 
        'ref_year': datetime.now().year, 
        'weekend_type': 'Standard (Sat-Sun)',
        'bimodal_found': []
    }
    
    # 1. 🕰️ AUTOMATIC TEMPORAL & BEHAVIORAL ALIGNMENT
    if date_cols:
        sample_date_col = date_cols[0]
        if sample_date_col in df.columns:
            temp_dates = pd.to_datetime(df[sample_date_col], errors='coerce')
            audit_report['ref_year'] = int(temp_dates.dt.year.max())
            
            day_counts = temp_dates.dt.weekday.value_counts()
            if 4 in day_counts and 5 in day_counts:
                # Friday activity vs Saturday activity
                if day_counts[4] > (day_counts[5] * 0.9):
                    audit_report['weekend_type'] = 'Extended (Fri-Sun)'

    # 2. 🤖 AUTO-BIMODAL DETECTION (The "Two-Hump" Fix)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col == target or '_per_' in col or '_squared' in col: continue
        
        # Detect if a significant "Zero-Universe" exists (10% to 90% zeros)
        zero_ratio = (df[col] == 0).sum() / len(df)
        if 0.10 < zero_ratio < 0.90:
            df[f'has_{col}_logic'] = (df[col] > 0).astype(int)
            audit_report['bimodal_found'].append(col)

    # 3. 🏗️ FEATURE CONSTRUCTION
    # Ratios: Value Density
    for a, b in ratio_pairs:
        if a in df.columns and b in df.columns: 
            df[f'{a}_per_{b}'] = df[a] / (df[b] + 1)
    
    # Curvature: Polynomials
    for c in poly_cols:
        if c in df.columns: 
            df[f'{c}_squared'] = df[c] ** 2
            
    # Temporal Deconstruction
    weekend_days = [4, 5, 6] if audit_report['weekend_type'] == 'Extended (Fri-Sun)' else [5, 6]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_weekday'] = df[col].dt.weekday
            df[f'{col}_is_weekend'] = df[col].dt.weekday.isin(weekend_days).astype(int)
            
    # Age/Tenure: Sync'd to the detected reference year
    for nn, c in age_map.items():
        if c in df.columns: 
            df[nn] = audit_report['ref_year'] - df[c]

    # 4. 📊 AUDIT GENERATION
    audit_report['counts'] = {
        'Ratios': len(ratio_pairs),
        'Polynomials': len(poly_cols),
        'Temporal': len(date_cols) * 4,
        'Age/Tenure': len(age_map),
        'Bimodal_Switches': len(audit_report['bimodal_found'])
    }
    
    return df, audit_report

def architecture_factory(df, target, phase=12, keep=20, model_obj=None):
    # Surgical Unpacking: If df is a tuple (df, report), take the dataframe
    if isinstance(df, tuple):
        df = df[0]
        
    if phase == 12:
        Xj = df.drop(columns=[target], errors='ignore').copy()
        for i in range(Xj.shape[1]):
            col_data = Xj.iloc[:, i]
            col_name = Xj.columns[i]
            if pd.api.types.is_datetime64_any_dtype(col_data) or col_data.dtype.kind == 'M':
                Xj[col_name] = col_data.astype(np.int64) // 10**9
                continue
            if col_data.dtype.kind in ['O', 'S'] or col_data.dtype.name == 'category':
                Xj[col_name] = col_data.astype('category').cat.codes
            Xj[col_name] = Xj[col_name].replace([np.inf, -np.inf], np.nan).fillna(-999).astype('float64')
        
        Xj = Xj.select_dtypes(include=[np.number])
        m = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1).fit(Xj, df[target])
        
        importance_df = pd.DataFrame({
            'feature': Xj.columns, 
            'importance': m.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        return importance_df, df

    if phase == 17:
        X_pipe = df.drop(columns=[target], errors='ignore')
        num_cols = X_pipe.select_dtypes(include=[np.number]).columns.tolist()
        
        # 🤖 AUTOMATED IMPUTATION STRATEGY
        avg_skew = X_pipe[num_cols].skew().mean()
        imp_strat = 'median' if abs(avg_skew) > 0.75 else 'mean'
        imp_reason = "data is skewed" if imp_strat == 'median' else "data is symmetric"
        
        # 🛡️ THE SANITIZER SUB-MACHINE
        pre = ColumnTransformer(transformers=[
            ('num', Pipeline([
                ('im', SimpleImputer(strategy=imp_strat)),
                ('pt', PowerTransformer()),
                ('ss', StandardScaler())
            ]), num_cols)
        ], n_jobs=1)
        
        # 🧠 THE INTELLIGENCE SWITCH
        is_regression = df[target].dtype.kind in 'if'
        
        if is_regression:
            final_model = TransformedTargetRegressor(
                regressor=model_obj, 
                func=np.log1p, 
                inverse_func=np.expm1
            )
            task_desc = f"Regression (Log-Transformed via np.log1p)"
        else:
            final_model = model_obj
            task_desc = "Classification (Standard Label Processing)"

        print("\n" + "—"*40)
        print(f"⚙️ PIPELINE AUTO-CONFIGURED:")
        print(f"🔹 IMPUTER : {imp_strat.upper()} ({imp_reason})")
        print(f"🔹 TASK    : {task_desc}")
        print("—"*40)

        return Pipeline([('preprocessor', pre), ('model', final_model)])
 
# --- BLOCK 4: THE AUDITOR (Phases 19, 20) ---
def execute_integrity_audit(df_final, model, target, winners=None):
    from sklearn.model_selection import train_test_split, cross_val_score
    
    # 1. Surgical Unpacking & Flattening
    if isinstance(df_final, tuple): df_final = df_final
    X = df_final.drop(columns=[target], errors='ignore')
    y_raw = df_final[target]
    y = y_raw.iloc[:, 0] if hasattr(y_raw, 'shape') and len(y_raw.shape) > 1 else y_raw
    
    # 2. Pipeline Sync
    audit_pipe = architecture_factory(df_final, target, phase=17, model_obj=model)
    
    # 3. THE STABILITY MATH (The 0.87 Check)
    print(f"🚀 Launching K-Fold Stability Audit for {type(model).__name__}...")
    cv_scores = cross_val_score(audit_pipe, X, y, cv=5, scoring='r2')
    
    # 4. DATA SPLIT for Residuals
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    audit_pipe.fit(Xt, yt)
    preds = audit_pipe.predict(Xv)
    
    # 5. MARKET LOGIC CHECK
    logic_pass = "N/A"
    if winners is not None:
        Xs = Xv.copy()
        top_feat = winners[0] if isinstance(winners, list) else winners
        Xs[top_feat] = Xs[top_feat] * 1.2
        logic_pass = "PASSED" if audit_pipe.predict(Xs).mean() > preds.mean() else "FAILED"

    # Return everything the Notebook Dashboard needs
    return {
        'y_test': yv,
        'y_preds': preds,
        'mean_r2': cv_scores.mean(),
        'std_r2': cv_scores.std(),
        'gap': abs(audit_pipe.score(Xt, yt) - audit_pipe.score(Xv, yv)),
        'logic': logic_pass
    }
