# USTC-matched v2 population (names-only, blinded)

Names-only inputs; no target column opened. Table: fr_group_alias.csv.

## (a) Alias merges (163 keys -> 148 identities; 7 merges)
- APP: app, ammonium polyphosphate, fr cros484.
- RDP: rdp, rdp resorcinol bis, resorcinol bis (truncated).
- AlPi: alpi, exolit op935, exolit op1230 (Clariant grades).
- boehmite: boehmite, boehmite alo, disperal 40, actilox b30.
- Mg(OH)2: magnesium hydroxide, mg2, mh, mw-mh2, ind-mh2 (mh unconfirmed).
- S600 (2 keys); UiO-66-NH2 (2 keys).
- Not merged: Exolit AP750 (formulated APP + N-synergist), modified APPs, coated red P, DOPO vs every derivative, DEPA-Al (a boehmite-phosphate hybrid; fr_chemistry wrongly says AlPi).
- Override: arg-app set to 0 (no P-C/P-O-C).
- Caveats: key `ifr` pools 6 different IFRs; actilox b30 / disperal 40 rows carry TAHHT-DDPO as 2nd FR, hidden by the first-FR key.

## (b) Proposed rules
1. EP, is_control==0. 2. material_form=='bulk'. 3. first FR organophosphorus==1 (76 keys, 72 identities).
Flags, not drops: A = nanofiller/filler/reinforcement/other component; B = second FR or synergist; H = hybrid FR; low = guessed chemistry.
Primary = rules 1-3; sensitivity drops A, B, then low and H.

## (c) Counts (non-control)
| rule | rows | papers | identities |
|---|---|---|---|
| EP | 500 | 122 | 148 |
| + bulk | 413 | 106 | 137 |
| + orgP (primary) | 168 | 53 | 65 |
| primary, conf>=med | 145 | 46 | 55 |
| primary, no A/B | 120 | 47 | 59 |
| no A/B, conf>=med, no H | 72 | 32 | 34 |

Flag A 22 rows, B 26, either 48. Bulk rule removes fibre 73, foam 5, film 3, other 6. Primary: 28 identities have one row; heat flux 50: 100 rows, 35: 55, 30: 10, 25: 3. Every primary paper has a bulk control.

## (d) Differences from USTC (USTC facts are abstract-level, unconfirmed)
- Source: ours is literature extraction from open-access papers; USTC's is unknown.
- Size: ~885 pHRR records claimed vs 168 here.
- Descriptors: ECFP4/MACCS need structures; only 7 of 76 organophosphorus keys have SMILES.
- Group definition: their FR molecule groups vs our name-based identity (unverified for unresolved acronyms).
- Heat flux scope unknown; ours is mixed 25-50.
- Neat-matrix input: ours is the same-paper control; hardener and thickness vary.
- Loading units mixed (wt%, phr, P%).
- Their scope for co-additives, hybrids, multi-FR rows unknown.

## (e) All-matrix alias (fr_group_alias_all.csv)
706 keys -> 634 identities (the 163 EP keys keep their identity text). Defaults: a key with no synonym is its own identity; unresolved acronyms are tagged "[unresolved]"; 85 rows are low confidence. Same-name abbreviations that are different chemicals were not merged (melamine pyrophosphate vs MPP, hccp vs HPCP, expanded vs expandable graphite, modified/coated APPs, Exolit RP 6500 vs RP 607).

Cross-matrix merges that touch EP keys (12 identities, 43 non-EP-file keys):
- AlPi (EP: alpi, exolit op935, exolit op1230) <- adp, aldpi, aluminium/aluminum diethyl(phosphinate) x4, aluminium phosphinate (OP1230).
- aluminium hypophosphite (EP: aluminum hypophosphite) <- ahp, ahpi, aluminium hypophosphite (low: that PA66 row is labelled Exolit OP935, possibly AlPi).
- ATH (EP: al3) <- ath, ath3), aluminium/aluminum hydroxide, aluminium trihydrate, al3 in surface film.
- Mg(OH)2 (EP: 5 keys) <- mdh, mg2 magnifin h10a, mhx/mhy/mhz, micro-sized mdh, nano-mg2.
- expandable graphite (EP: eg) <- expandable graphite, eg only, untreated eg, eg1, eg2, 3772, eg1-3 and GHL PX 95 HT grades (not expanded graphite, not treated/hydrothermal EG).
- HPCP (EP: hexaphenoxycyclotriphosphazene) <- hpctp, hexaphenoxycyclotriphosphazene hpctp.
- MPP (EP: mpp) <- melamine polyphosphate. PPAP (EP: ppap) <- papp, piperazine pyrophosphate.
- black phosphorus (EP: few-layer) <- black phosphorus nanosheets (medium).
- ZIF-8 <- zeolitic imidazolate framework zif-8. MgAlZn-LDH-CO3 <- znmgal-co3-ldh. aminopropylphenyl silsesquioxane <- polysilsesquioxane.
