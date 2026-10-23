# Cone-calorimeter literature rows (multi-property), started 2026-10-11 (Claude, Arnav's session)

Purpose: a separate, new literature dataset of fire properties of flame-retarded polymers and foams (TTI, pHRR, THR, residue, and LOI where reported). It is NOT part of the FSRI-based records (pHRR/TTI/THR/CR) and not part of the LOI benchmark; it has no MCC/STA inputs. Whether and how it is used for modelling is a human/orchestrator decision.
Source papers: PDFs in ~/Dravix/Data (outside the repo; Las Positas College library full texts and user-supplied PDFs). Extraction by delegated agents from the pdftotext text, one batch per agent (batches/), values copied from tables/explicit text only; figure-only values never extracted; every value carries an evidence snippet. Independent QA: not done unless stated in QA_*.md.

Schema (rows_cN.csv): row_id, pdf_stem, doi, polymer_system, formulation_label, fr_summary, fr_loading_as_printed, heat_flux_kWm2, thickness_mm, density_kgm3, tti_s, phrr_kWm2, time_to_phrr_s, thr_MJm2, avg_hrr_kWm2, tsr_m2m2, char_residue_pct, loi_pct, ul94, extraction (table|text), evidence, notes
