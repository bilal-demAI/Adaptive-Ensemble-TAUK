import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None
from sklearn.model_selection import GroupKFold

from .config import AdaptiveConfig
from .utils import set_seed, get_device, ce_loss, tune_threshold_for_uar, eval_all, oversample, majority_hard_from_probs, extract_ids
from .embeddings import embed_texts_transformer, embed_audio_whisper, embed_audio_egemaps
from .models import make_base_pool, predict_proba_pos
from .oof import oof_preds_all_models, fit_meta_with_inner_oof, make_group_splitter
from .router import (select_pair_expected_gain, ranks_from_probs, pairwise_rmsd, pairwise_zol_disagreement, build_router_features, 
    platt_fit_apply, best_alpha, best_alpha_for_uar,select_pair_oracle_uar_gain,isotonic_fit_apply
)
from .moe import fit_moe_gate, moe_predict, gate_degeneracy
from .viz import plot_decision_boundary_2d, find_hard_indices
from .strategies import build_strategy_predictor
from .embeddings import (
    get_or_compute_text_embeddings,
    get_or_compute_whisper_embeddings,
    get_or_compute_egemaps_embeddings,
    get_or_compute_openai_embeddings,
)
import os
import numpy as np
import pandas as pd
import torch
import itertools


def _build_feature_spaces(df: pd.DataFrame, cfg: AdaptiveConfig, device=None) -> Dict[str, np.ndarray]:
    device = device or get_device()
    texts = df[cfg.text_col].astype(str).str.strip().tolist()
    audio_paths = df[cfg.audio_col].tolist() if (cfg.audio_col and cfg.audio_col in df.columns) else None

    # IMPORTANT: stable per-row IDs
    ids = extract_ids(df, key_col=getattr(cfg, "key_col", None), audio_col=getattr(cfg, "audio_col", None))

    spaces = {}
    if "roberta" in cfg.features:
        spaces["roberta"] = get_or_compute_text_embeddings(
            texts, cfg.roberta_model, cfg.max_length, cfg.batch_size, True,
            ids=ids, dataset_tag="train", feat_name="roberta", cfg=cfg
        )
    if "deberta" in cfg.features:
        spaces["deberta"] = get_or_compute_text_embeddings(
            texts, cfg.deberta_model, cfg.max_length, cfg.batch_size, True,
            ids=ids, dataset_tag="train", feat_name="deberta", cfg=cfg
        )
    if "whisper" in cfg.features:
        if audio_paths is None:
            print("[Whisper] audio_col not provided/found; skipping Whisper.")
        else:
            spaces["whisper"] = get_or_compute_whisper_embeddings(
                audio_paths, cfg.whisper_model, batch_size=max(1, cfg.batch_size//2),
                ids=ids, dataset_tag="train", cfg=cfg
            )
    if ("openai" in cfg.features) and cfg.use_openai_embed:
        spaces["openai"] = get_or_compute_openai_embeddings(
            texts, ids=ids, dataset_tag="train", cfg=cfg
        )

    if ("egemaps" in cfg.features) and cfg.use_egemaps:
        if audio_paths is None:
            print("[eGeMAPS] audio_col not provided/found; skipping eGeMAPS.")
        else:
            spaces["egemaps"] = get_or_compute_egemaps_embeddings(
                audio_paths,
                feature_set=cfg.opensmile_feature_set,
                feature_level=cfg.opensmile_feature_level,
                proj_dim=cfg.opensmile_proj_dim,
                rff_gamma=cfg.opensmile_rff_gamma,
                seed=cfg.seed,
                ids=ids, dataset_tag="train", cfg=cfg
            )

    if cfg.concat_features and len(spaces) >= 2:
        spaces["concat"] = np.concatenate([spaces[k] for k in sorted(spaces.keys())], axis=1)
    return spaces
def _calib_apply_for_model(name, p_tr, y_tr, p_va):
    if any(tag in name.lower() for tag in ["rf","xgb","gnb"]):
        return isotonic_fit_apply(p_tr, y_tr, p_va)
    else:
        return platt_fit_apply(p_tr, y_tr, p_va)


def run_adaptive_pipeline(cfg: AdaptiveConfig):
    set_seed(cfg.seed)
    device = get_device()
    df = pd.read_csv(cfg.csv_path).dropna(subset=[cfg.text_col]).reset_index(drop=True)
    from .utils import extract_ids
    ids_all = extract_ids(df, key_col=getattr(cfg, "key_col", None), audio_col=getattr(cfg, "audio_col", None))



    raw_labels = df[cfg.label_col].astype(str).str.strip()
    # lock mapping to maintain label order if you publish (replace with your map if needed)
    label_map = {'NC': 0, 'MCI': 1} if set(raw_labels.unique()) == set(['NC','MCI']) else {lab:i for i, lab in enumerate(sorted(raw_labels.unique()))}
    print("Label mapping:", label_map)
    y = raw_labels.map(label_map).values.astype(int)

    groups = None
    if cfg.group_col and cfg.group_col in df.columns:
        groups = df[cfg.group_col].astype(str).values
        print("[CV] Subject-wise grouping ON")

    feat_spaces = _build_feature_spaces(df, cfg, device=device)
    results = {}

    # build outer CV iterator
    if groups is not None:
        SplitterCls, kwargs = make_group_splitter(cfg.n_splits, seed=cfg.seed, stratified=True)
        fold_iter = list(SplitterCls(**kwargs).split(X=np.zeros(len(y)), y=y, groups=groups))
    else:
        if cfg.use_repeated_cv:
            splitter = RepeatedStratifiedKFold(n_splits=cfg.n_splits, n_repeats=cfg.cv_repeats, random_state=cfg.seed)
        else:
            splitter = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        fold_iter = list(splitter.split(np.zeros(len(y)), y))

    for feat_name, X in feat_spaces.items():
        print(f"\n=== Feature space: {feat_name} | X={X.shape} ===")
        fold_metrics = {
            "single": {},
            "mv_soft": [], "mv_hard": [],
            "stack_lr": [], "stack_xgb": [],
            "adaptive_router": [],
            "moe_soft": [], "moe_hard": [],
            "distill_student": [],
        }
        for fold, (tr_idx, va_idx) in enumerate(fold_iter):
            print(f"\n[Fold {fold}]")
            groups_tr = groups[tr_idx] if groups is not None else None

            ids_tr = ids_all[tr_idx]
            ids_va = ids_all[va_idx]

            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

            if feat_name == "egemaps" and getattr(cfg, "egemaps_fold_project", False):
                from sklearn.pipeline import make_pipeline
                steps = [StandardScaler(with_mean=True)]
                if cfg.opensmile_proj_dim and cfg.opensmile_proj_dim < X_tr.shape[1]:
                    steps += [PCA(n_components=cfg.opensmile_proj_dim, random_state=cfg.seed)]
                scaler_pca = make_pipeline(*steps)
                X_tr = scaler_pca.fit_transform(X_tr)
                X_va = scaler_pca.transform(X_va)

            # base pool
            X_tr_os, y_tr_os = oversample(X_tr, y_tr, target_ratio=cfg.oversample_ratio)
            ratio = (y_tr_os == 0).sum() / max(1, (y_tr_os == 1).sum())
            pool = make_base_pool(ratio, pca_dim=cfg.pca_dim, calibrate_trees=cfg.calibrate_trees)
            for clf in pool.values(): clf.fit(X_tr_os, y_tr_os)

            # init singles bucket once
            if not fold_metrics["single"]:
                for name in pool.keys(): fold_metrics["single"][name] = []

            preds_va = {name: predict_proba_pos(clf, X_va) for name, clf in pool.items()}

            # singles
            single_res = {}
            for name, p in preds_va.items():
                thr, _ = tune_threshold_for_uar(y_va, p)
                res = eval_all(y_va, p, thr)
                single_res[name] = res
                fold_metrics["single"][name].append(res)
            print("  Singles:", " | ".join([f"{n}: F1 {single_res[n]['F1']:.3f}/UAR {single_res[n]['UAR']:.3f}" for n in single_res.keys()]))

            # --- Quality floor for candidate bases (protect UAR)
            model_list = list(pool.keys())
            # Choose your criterion: "ROC_AUC" is stable; "UAR" is ok too.
            crit = getattr(cfg, "pair_quality_metric", "ROC_AUC")
            K = int(getattr(cfg, "pair_quality_topk", max(2, int(np.ceil(len(model_list) * 0.6)))))
            
            scores = {name: single_res[name][crit] for name in model_list}
            # Pick top-K by quality
            topk_models = [name for name, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:K]]
            if len(topk_models) < 2:
                topk_models = model_list  # fallback
            
            # Use topk_models in diversity/gain selection below
            candidates = topk_models

            

            # MV soft
            mat = np.column_stack([preds_va[k] for k in pool.keys()])
            avg = mat.mean(axis=1)
            thr_avg, _ = tune_threshold_for_uar(y_va, avg)
            fold_metrics["mv_soft"].append(eval_all(y_va, avg, thr_avg))

            # MV hard
            thr_vec = np.array([tune_threshold_for_uar(y_va, preds_va[k])[0] for k in pool.keys()])
            vote_frac = majority_hard_from_probs(mat, thr_vec)
            thr_hard, _ = tune_threshold_for_uar(y_va, vote_frac)
            fold_metrics["mv_hard"].append(eval_all(y_va, vote_frac, thr_hard))

            # stacking
            meta_LR, _  = fit_meta_with_inner_oof(X_tr, y_tr, pool, model_list, n_inner=5, meta_type="lr",
                                                  oversample_ratio=cfg.oversample_ratio, pca_dim=cfg.pca_dim, calibrate_trees=cfg.calibrate_trees,
                                                  groups_tr=groups_tr)
            meta_XGB, _ = fit_meta_with_inner_oof(X_tr, y_tr, pool, model_list, n_inner=5, meta_type="xgb",
                                                  oversample_ratio=cfg.oversample_ratio, pca_dim=cfg.pca_dim, calibrate_trees=cfg.calibrate_trees,
                                                  groups_tr=groups_tr)
            meta_X_va = np.column_stack([preds_va[n] for n in model_list])
            p_stack_lr  = meta_LR.predict_proba(meta_X_va)[:,1] if hasattr(meta_LR, "predict_proba") else meta_LR.predict(meta_X_va)
            p_stack_xgb = meta_XGB.predict_proba(meta_X_va)[:,1] if hasattr(meta_XGB, "predict_proba") else meta_XGB.predict(meta_X_va)
            t_lr,_ = tune_threshold_for_uar(y_va, p_stack_lr)
            t_xg,_ = tune_threshold_for_uar(y_va, p_stack_xgb)
            fold_metrics["stack_lr"].append(eval_all(y_va, p_stack_lr, t_lr))
            fold_metrics["stack_xgb"].append(eval_all(y_va, p_stack_xgb, t_xg))

            # --- Adaptive Router ---
            cfg.router_uncertainty_eps = getattr(cfg, "router_uncertainty_eps", 0.08)   # |pr-0.5| < 0.08 → uncertain
            cfg.router_uncertainty_fallback = getattr(cfg, "router_uncertainty_fallback", "mv_hard")
            
            P_oof, full_pool = oof_preds_all_models(X_tr, y_tr, model_list, n_inner=5,
                oversample_ratio=cfg.oversample_ratio, pca_dim=cfg.pca_dim, calibrate_trees=cfg.calibrate_trees, groups_tr=groups_tr)
            # Calibrate OOF to TRUE labels (used for relabeling and weights)
            P_cal = np.zeros_like(P_oof)
            for i in range(P_oof.shape[1]):
                P_cal[:, i] = platt_fit_apply(P_oof[:, i], y_tr, P_oof[:, i])
            
            # Validation predictions for ALL models (class-calibrated to TRUE labels)
            P_val_raw = np.column_stack([predict_proba_pos(full_pool[name], X_va) for name in model_list])
            P_val_cal = np.column_stack([platt_fit_apply(P_oof[:, i], y_tr, P_val_raw[:, i]) for i in range(P_oof.shape[1])])

            if cfg.pair_selection.lower() == "gain":
                # Use calibrated OOF for expected gain (per-sample loss available)
                (mA, mB), _ = select_pair_expected_gain(P_cal, y_tr, model_list, metric="logloss")
                print(f"  Pair by expected gain: {mA}+{mB}")
            elif cfg.pair_selection.lower() == "gain_uar":
                (mA, mB), g = select_pair_oracle_uar_gain(P_cal, y_tr, model_list, candidates)
                print(f"  Pair by oracle UAR gain: {mA}+{mB} (ΔUAR={g:.4f})")
            else:
                # Diversity by RMSD of ranks on VALIDATION (paper-aligned) over 'candidates'
                best_pair, best_d = None, -1.0
                for a, b in itertools.combinations(candidates, 2):
                    ia, ib = model_list.index(a), model_list.index(b)
                    ra = ranks_from_probs(P_val_cal[:, ia], ascending=True).astype(float)
                    rb = ranks_from_probs(P_val_cal[:, ib], ascending=True).astype(float)
                    d = pairwise_rmsd(ra, rb)
                    if d > best_d:
                        best_d, best_pair = d, (a, b)
                mA, mB = best_pair
                print(f"  Pair by RMSD: {mA}+{mB} (RMSD={best_d:.2f})")

            pA_oof = P_cal[:, model_list.index(mA)]
            pB_oof = P_cal[:, model_list.index(mB)]
            rA = ranks_from_probs(pA_oof, True)
            rB = ranks_from_probs(pB_oof, True)
            
            # A if positive & rA>=rB OR negative & rA<=rB; else B
            choose_A = ((y_tr == 1) & (rA >= rB)) | ((y_tr == 0) & (rA <= rB))
            relabels = np.where(choose_A, 0, 1)
            
            # tie-break where rA==rB: prefer the model with larger |p-0.5| (more decisive)
            ties = (rA == rB)
            if ties.any():
                prefer_A = np.abs(pA_oof - 0.5) >= np.abs(pB_oof - 0.5)
                relabels[ties] = np.where(prefer_A[ties], 0, 1)
            
            skew = relabels.mean()
            print(f"  Router label skew: {skew:.3f}")

            # Sample weights for router (optional but helpful on imbalanced data)
            if cfg.router_weight == "delta_ce":
                la = ce_loss(y_tr, pA_oof)   # per-sample vectors
                lb = ce_loss(y_tr, pB_oof)
                w = np.abs(la - lb)
                oracle_router = np.minimum(la, lb).mean()
                best_single  = min(la.mean(), lb.mean())
                print(f"  Oracle routing CE: {oracle_router:.4f} vs best single CE: {best_single:.4f}")
            elif cfg.router_weight == "margin":
                w = np.abs(pA_oof - pB_oof)
            else:
                w = None
            # UAR boost: upweight minority TRUE class a bit (1.5–3.0 usually safe)
            pos_w = float(getattr(cfg, "router_pos_weight", 2))
            w = w * np.where(y_tr == 1, pos_w, 1.0)

            # CLASS-calibrated validation probs for final prediction stream
            pA_va_cls = _calib_apply_for_model(mA, pA_oof, y_tr, predict_proba_pos(full_pool[mA], X_va))
            pB_va_cls = _calib_apply_for_model(mB, pB_oof, y_tr, predict_proba_pos(full_pool[mB], X_va))
            
            # Fallback if router labels are too skewed (can hurt UAR)
            if skew < 0.25 or skew > 0.75:
                # Choose blend on VALIDATION to favor UAR downstream
                alpha = (best_alpha_for_uar(y_va, pA_va_cls, pB_va_cls)
                         if getattr(cfg, "fallback_optimize", "uar").lower() == "uar"
                         else best_alpha(y_va, pA_va_cls, pB_va_cls))
                p_adapt = alpha * pA_va_cls + (1 - alpha) * pB_va_cls
            else:
                # Router trained on ORIGINAL features (paper-aligned)
                # Router trained on ORIGINAL features (+1 confidence feature)
                router_X_tr = np.column_stack([X_tr, np.abs(pA_oof - 0.5) - np.abs(pB_oof - 0.5)]) \
                              if getattr(cfg, "router_add_margin_feature", True) else X_tr
                router_X_va = np.column_stack([X_va, np.abs(pA_va_cls - 0.5) - np.abs(pB_va_cls - 0.5)]) \
                              if getattr(cfg, "router_add_margin_feature", True) else X_va
                
                # class ratio for relabels
                rel_pos = max(1, int(relabels.sum()))
                rel_neg = max(1, int((relabels == 0).sum()))
                spw = rel_neg / rel_pos
            
                if cfg.router_meta == "xgb":
                    from xgboost import XGBClassifier
                    print("__Using XGB-adaptive_router__")
                    router = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.07,
                                           subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42, scale_pos_weight=spw)
                elif cfg.router_meta == "rf":
                    print("__Using RF-adaptive_router__")
                    from sklearn.ensemble import RandomForestClassifier
                    router = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42,
                                                    class_weight="balanced", n_jobs=-1)
                else:
                    print("__Using LR-adaptive_router__")
                    router = LogisticRegression(solver="lbfgs", max_iter=2000,
                                                class_weight="balanced", C=0.5)
                router.fit(router_X_tr, relabels, sample_weight=w)

                def _uar_from_scores(y_true, p_scores):
                    thr, _ = tune_threshold_for_uar(y_true, p_scores)
                    return eval_all(y_true, p_scores, thr)["UAR"]
                
                if cfg.confidence_blend and hasattr(router, "predict_proba"):
                    # SOFT gate: tune tau for UAR on this fold's validation
                    pr = router.predict_proba(router_X_va)[:, 1]
                
                    # Try a modest grid; widen if you like
                    tau_grid = getattr(cfg, "tau_grid", [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 25])
                    best_uar, best_p = -1.0, None
                    for tau in tau_grid:
                        lam = 1.0 / (1.0 + np.exp(-tau * (pr - 0.5)))  # [0,1]
                        p_try = (1 - lam) * pA_va_cls + lam * pB_va_cls
                        uar = _uar_from_scores(y_va, p_try)
                        if uar > best_uar:
                            best_uar, best_p = uar, p_try
                    p_adapt = best_p
                else:
                    # HARD gate: tune router threshold theta for UAR on validation
                    if hasattr(router, "predict_proba"):
                        pr = router.predict_proba(router_X_va)[:, 1]
                        theta_grid = getattr(cfg, "router_theta_grid", np.linspace(0.1, 0.9, 17))
                        best_uar, best_p = -1.0, None
                        for th in theta_grid:
                            route = (pr >= th).astype(int)  # 1 => choose B
                            p_try = np.where(route == 0, pA_va_cls, pB_va_cls)
                            uar = _uar_from_scores(y_va, p_try)
                            if uar > best_uar:
                                best_uar, best_p = uar, p_try
                        p_adapt = best_p
                    else:
                        # fallback to predict(); then still tune post-threshold as you already do
                        route = router.predict(router_X_va)
                        p_adapt = np.where(route == 0, pA_va_cls, pB_va_cls)
                        


            t_ad, _ = tune_threshold_for_uar(y_va, p_adapt)
            fold_metrics["adaptive_router"].append(eval_all(y_va, p_adapt, t_ad))
            # Post-router headroom realization (just for visibility)
            thr_bestA, _ = tune_threshold_for_uar(y_va, pA_va_cls)
            thr_bestB, _ = tune_threshold_for_uar(y_va, pB_va_cls)
            uar_best_single = max(eval_all(y_va, pA_va_cls, thr_bestA)["UAR"],
                                  eval_all(y_va, pB_va_cls, thr_bestB)["UAR"])
            thr_ad, _ = tune_threshold_for_uar(y_va, p_adapt)
            uar_ad = eval_all(y_va, p_adapt, thr_ad)["UAR"]
            print(f"  UAR best_single_val={uar_best_single:.3f} | adaptive_val={uar_ad:.3f}")

            # --- MoE ---
            if cfg.include_moe:
                P_val = np.column_stack([predict_proba_pos(full_pool[name], X_va) for name in model_list])
                gate = fit_moe_gate(P_oof, y_tr, hidden=0, lr=5e-3, epochs=300, temperature=1.0, entropy_reg=1e-3, patience=30, device=device)
                with torch.no_grad():
                    w_check = gate(torch.tensor(P_oof[:min(64, len(P_oof))], dtype=torch.float32, device=device))
                collapsed = gate_degeneracy(w_check, thresh=0.10)
                if collapsed:
                    p_soft = P_val.mean(axis=1)
                else:
                    p_soft = moe_predict(gate, P_val, hard=False, device=device)
                ts,_ = tune_threshold_for_uar(y_va, p_soft)
                fold_metrics["moe_soft"].append(eval_all(y_va, p_soft, ts))

                if collapsed:
                    aucs = [roc_auc_score(y_tr, P_oof[:,i]) for i in range(P_oof.shape[1])]
                    best_idx = int(np.nanargmax(aucs))
                    p_hard = P_val[:, best_idx]
                else:
                    p_hard = moe_predict(gate, P_val, hard=True, device=device)
                th,_ = tune_threshold_for_uar(y_va, p_hard)
                fold_metrics["moe_hard"].append(eval_all(y_va, p_hard, th))

            # --- Distillation (teacher -> student) ---
            if cfg.include_distill:
                print("  [KD] teacher:", cfg.distill_teacher)
                P_val = np.column_stack([predict_proba_pos(full_pool[name], X_va) for name in model_list])
                teacher = cfg.distill_teacher.lower()
                if teacher in ("mv_soft", "avg_all"):
                    q_oof = P_oof.mean(axis=1); q_val = P_val.mean(axis=1)
                elif teacher == "mv_hard":
                    thr_vec = np.array([tune_threshold_for_uar(y_tr, P_oof[:,i])[0] for i in range(P_oof.shape[1])])
                    q_oof = (P_oof >= thr_vec.reshape(1, -1)).astype(int).mean(axis=1)
                    q_val = (P_val >= thr_vec.reshape(1, -1)).astype(int).mean(axis=1)
                elif teacher in ("stack_lr","stack_xgb"):
                    meta_type = "lr" if teacher=="stack_lr" else "xgb"
                    meta, _ = fit_meta_with_inner_oof(X_tr, y_tr, pool, model_list, n_inner=5, meta_type=meta_type,
                        oversample_ratio=cfg.oversample_ratio, pca_dim=cfg.pca_dim, calibrate_trees=cfg.calibrate_trees, groups_tr=groups_tr)
                    q_oof = meta.predict_proba(P_oof)[:,1] if hasattr(meta,"predict_proba") else meta.predict(P_oof)
                    q_val = meta.predict_proba(P_val)[:,1] if hasattr(meta,"predict_proba") else meta.predict(P_val)
                elif teacher.startswith("moe_"):
                    hard = (teacher=="moe_hard")
                    gate = fit_moe_gate(P_oof, y_tr, hidden=0, lr=5e-3, epochs=300, temperature=1.0, entropy_reg=1e-3, patience=30, device=device)
                    q_oof = moe_predict(gate, P_oof, hard=hard, device=device)
                    q_val = moe_predict(gate, P_val, hard=hard, device=device)
                elif teacher == "adaptive_router":
                    # reuse router logic for teacher
                    pA_oof = pA_oof; pB_oof = pB_oof  # already computed above
                    router_X_tr = build_router_features(pA_oof, pB_oof, use_ranks=cfg.use_rank_features)
                    from sklearn.ensemble import RandomForestClassifier
                    router_t = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42,
                                                      class_weight="balanced", n_jobs=-1)
                    router_t.fit(router_X_tr, relabels)
                    q_oof = np.where(router_t.predict(router_X_tr)==0, pA_oof, pB_oof)
                    pA_va = platt_fit_apply(pA_oof, relabels, predict_proba_pos(full_pool[mA], X_va))
                    pB_va = platt_fit_apply(pB_oof, relabels, predict_proba_pos(full_pool[mB], X_va))
                    q_val = np.where(router_t.predict(build_router_features(pA_va, pB_va, use_ranks=cfg.use_rank_features))==0, pA_va, pB_va)
                elif teacher == "best_single":
                    aucs = [roc_auc_score(y_tr, P_oof[:,i]) for i in range(P_oof.shape[1])]
                    idx = int(np.nanargmax(aucs)); q_oof = P_oof[:, idx]; q_val = P_val[:, idx]
                else:
                    raise ValueError(f"Unknown distill_teacher={teacher}")

                # oversample coherently for KD
                Xtr_for_kd, ytr_for_kd, qtr_for_kd = X_tr, y_tr, q_oof
                if cfg.distill_oversample:
                    Z = np.column_stack([X_tr, q_oof.reshape(-1,1)])
                    Z_os, y_os = oversample(Z, y_tr, target_ratio=cfg.oversample_ratio)
                    Xtr_for_kd = Z_os[:, :-1]; qtr_for_kd = Z_os[:, -1]; ytr_for_kd = y_os

                from .distill import train_kd_student
                _, _, student_predict = train_kd_student(
                    Xtr_for_kd, ytr_for_kd, qtr_for_kd,
                    X_va, y_va, q_val,
                    hidden=cfg.distill_hidden, alpha=cfg.distill_alpha, T=cfg.distill_temperature,
                    lr=cfg.distill_lr, wd=cfg.distill_weight_decay,
                    batch_size=cfg.distill_batch_size, epochs=cfg.distill_epochs,
                    patience=cfg.distill_patience, device=device
                )
                p_student = student_predict(X_va)
                tsd,_ = tune_threshold_for_uar(y_va, p_student)
                fold_metrics["distill_student"].append(eval_all(y_va, p_student, tsd))

            # ----- Generic per-fold decision-boundary plotting -----
            if getattr(cfg, "plot_enable", False):
                os.makedirs(cfg.plot_output_dir, exist_ok=True)
            
                for strat_to_plot in getattr(cfg, "plot_strategies"):
                    try:
                        # Build predictor for the requested strategy on this fold's TRAIN
                        predictor, thr_plot, _, full_pool_fold = build_strategy_predictor(
                            X_tr, y_tr,
                            model_list=list(pool.keys()),
                            strategy=strat_to_plot,
                            cfg=cfg,
                            groups_tr=groups_tr if 'groups_tr' in locals() else None
                        )
            
                        # Optionally re-tune threshold on the split we’re about to plot
                        if cfg.plot_on in ("train", "both"):
                            p_train_plot = predictor(X_tr)
                            thr_train_plot, _ = tune_threshold_for_uar(y_tr, p_train_plot)
                            save_path = os.path.join(cfg.plot_output_dir, f"{feat_name}_fold{fold}_{strat_to_plot}_train.png")
                            plot_decision_boundary_2d(
                                X=X_tr, y=y_tr,
                                predict_proba_fn=predictor,
                                thr=thr_train_plot,
                                title=f"{feat_name} fold {fold} | {strat_to_plot} (TRAIN)",
                                base_pool=full_pool_fold,
                                save_path=save_path,
                                grid_n=getattr(cfg, "plot_grid_n", 300),
                                highlight_hard_k=getattr(cfg, "plot_hard_k", 50)
                            )
                            # Save hard examples for TRAIN
                            hard_idx, _ = find_hard_indices(y_tr, p_train_plot, thr_train_plot, top_k=getattr(cfg, "plot_hard_k", 50))
                            pd.DataFrame({
                                "index": ids_tr[hard_idx],
                                "prob": p_train_plot[hard_idx],
                                "margin": np.abs(p_train_plot[hard_idx] - thr_train_plot),
                                "split": "train",
                                "strategy": strat_to_plot,
                                "feature": feat_name,
                                "fold": fold
                            }).to_csv(os.path.join(cfg.plot_output_dir, f"{feat_name}_fold{fold}_{strat_to_plot}_hard_train.csv"), index=False)
            
                        if cfg.plot_on in ("val", "both"):
                            p_val_plot = predictor(X_va)
                            thr_val_plot, _ = tune_threshold_for_uar(y_va, p_val_plot)
                            save_path = os.path.join(cfg.plot_output_dir, f"{feat_name}_fold{fold}_{strat_to_plot}_val.png")
                            plot_decision_boundary_2d(
                                X=X_va, y=y_va,
                                predict_proba_fn=predictor,
                                thr=thr_val_plot,
                                title=f"{feat_name} fold {fold} | {strat_to_plot} (VAL)",
                                base_pool=full_pool_fold,
                                save_path=save_path,
                                grid_n=getattr(cfg, "plot_grid_n", 300),
                                highlight_hard_k=getattr(cfg, "plot_hard_k", 50)
                            )
                            # Save hard examples for VAL
                            hard_idx, _ = find_hard_indices(y_va, p_val_plot, thr_val_plot, top_k=getattr(cfg, "plot_hard_k", 50))
                            pd.DataFrame({
                                "index_in_val": ids_va[hard_idx],
                                "prob": p_val_plot[hard_idx],
                                "margin": np.abs(p_val_plot[hard_idx] - thr_val_plot),
                                "true": y_va[hard_idx],
                                "pred": (p_val_plot[hard_idx] >= thr_val_plot).astype(int),
                                "split": "val",
                                "strategy": strat_to_plot,
                                "feature": feat_name,
                                "fold": fold
                            }).to_csv(os.path.join(cfg.plot_output_dir, f"{feat_name}_fold{fold}_{strat_to_plot}_hard_val.csv"), index=False)

                    except Exception as e:
                        print(f"[viz] skipping {strat_to_plot} for {feat_name}/fold{fold}: {e}")

        # ---- summarize (mean ± std) ----
        def summarize(metric_list):
            keys = metric_list[0].keys()
            return {k: (float(np.nanmean([m[k] for m in metric_list])), float(np.nanstd([m[k] for m in metric_list])))
                    for k in keys}

        summary = {}
        for name, lst in fold_metrics["single"].items():
            summary[f"best_single_{name}"] = summarize(lst)
        for k in ["mv_soft","mv_hard","stack_lr","stack_xgb","adaptive_router","moe_soft","moe_hard","distill_student"]:
            if fold_metrics[k]:
                summary[k] = summarize(fold_metrics[k])

        results[feat_name] = {"fold_metrics": fold_metrics, "summary": summary}
        print("\n=== CV Summary (mean ± std) ===")
        for name, summ in summary.items():
            F1m,F1s = summ["F1"]; UARm,UARs = summ["UAR"]; SENm,SENs = summ["SEN"]; SPECm,SPECs = summ["SPEC"]
            ROCm,ROCs = summ["ROC_AUC"]; PRm,PRs = summ["PR_AUC"]
            print(f"{name:18s} | F1 {F1m:.3f}±{F1s:.3f} | UAR {UARm:.3f}±{UARs:.3f} | SEN {SENm:.3f}±{SENs:.3f} | "
                  f"SPEC {SPECm:.3f}±{SPECs:.3f} | ROC_AUC {ROCm:.3f}±{ROCs:.3f} | PR_AUC {PRm:.3f}±{PRs:.3f}")

    return {"results": results, "df": df, "feature_shapes": {k:v.shape for k,v in feat_spaces.items()}}
