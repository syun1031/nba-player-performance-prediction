# 上季球員表現 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans in this session; final fresh read-only review.

**Goal:** 核對並加入上季五項特徵，依序比較原九項與1＋3＋同區＋2倍方案。

**Architecture:** 獨立特徵模組、8fit執行器與唯讀驗證器，沿用v6共同樣本與v7對照，另存v9。

**Tech Stack:** 現有Python、NumPy、XGBoost、joblib、unittest。

**Spec:** docs/superpowers/specs/2026-10-30-prior-season-profile-design.md

## Global Constraints

不覆寫舊成果、不重評三月、不換線上。上一季2024-10-22～2025-04-13，只計正分鐘；本季截止2026-02-28。共同13221訓練，三折4198/8302/10724及4104/2422/2497，CV9023與Feb3462。8fit固定。

## Review Focus

跨隊同名資料加總而非場均再平均；缺資料不是零分或新秀標記；日期跨季錯放需拒絕；上季0分鐘不入統計、0分保留；推論不讀當場標籤，權重依本折原PTS。

### Task 1 特徵與資料核對

Create scripts/prior_season_profile.py and tests/test_prior_season_profile.py. Interfaces: build_profiles(rows)->dict keyed by exact PLAYER; augment(rows,profiles)->new rows; profile_matrix(rows,columns)->float matrix. Five feature names follow spec.

- [x] 寫測試覆蓋跨隊合併／總分總分鐘、無歷史NaN、日期／重複／無效值、零分鐘與零分、來源不被改寫、推論不讀標籤，先看失敗再實作至通過。
- [x] 執行既有2024–25 verify_csv.py，保存核對及484人接合名單。

### Task 2 固定實驗

Create scripts/experiment_prior_season_profile.py. Variants basic_prior14 multiplier1; combined_prior25_w2 multiplier2; XGBRegressor direct target_pts; same frozen params. Save features/train.csv,validation.csv, player_profiles.csv, player_join_audit.csv; models,fit_log,predictions,scores,alerts,8paired intervals,report,validation hashes. Refuse output overwrite before fit.

- [x] 預存協議與來源／舊成果雜湊，逐折兩候選，重載核對、推論去標籤及假標籤一致，嚴格8fit。
- [x] 重用v7對照並驗證鍵／標籤一致，保存完整結果與高分代價，不憑微小差異宣稱改善。

### Task 3 獨立驗證與結案

Create scripts/verify_prior_season_profile.py. Recompute profile sums with independent arithmetic from raw, feature joins/missingness, train-only weights, eight saved models, unweighted scores/alerts/AP and hashes. No fit.

- [x] 執行唯讀驗證與完整unittest；fresh reviewer檢查後修正實質問題。
- [x] 更新TODO/PROGRESS/PRD/README，保存review_and_tests與independent_audit，不部署。

## Execution ledger

Ruling: 使用者已授權照建議實作且追加組合，直接完成，不再要求重複確認。現有未追蹤研究程式與成果是本輪依賴；在原目錄新增獨立檔與v9成果，保護雜湊，不搬動或提交使用者既有改動。未另建缺少這些依賴的worktree；不commit/push。

Task 1: complete — 7 tests RED→GREEN; raw source verification passed.
Task 2: complete — exactly 8 fits; common cohort and frozen controls preserved.
Task 3: complete — 176 tests passed; independent reviewer found no required fixes; no deployment.
