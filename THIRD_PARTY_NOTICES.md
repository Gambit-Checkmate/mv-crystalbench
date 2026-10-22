# Third-party materials

MV-CrystalBench source code is licensed under MIT. Its generated question
annotations and rendered benchmark images are licensed under CC BY 4.0; see
`LICENSE-DATA`. These grants do not replace the licenses of upstream material.

## AFLOW prototype encyclopedia

`data/structure_sources/aflow/` includes an unmodified subset of the AFLOW
prototype encyclopedia CIF files, `info.json` descriptions and `base.bib`
references at revision `fb7421bb2522888b6a0a4ff91e124c6c65191096`.

Source: <https://github.com/aflow-org/aflow_prototype_encyclopedia>

Copyright 2026 M.J. Mehl, H. Eckert, S. Curtarolo. This material is licensed
under Apache License 2.0. The original `LICENSE`, `NOTICE`, and prototype-level
bibliographies are retained in that directory. MV-CrystalBench's bond policies,
selection manifest and scientific tests are project additions, not AFLOW
annotations. See the retained NOTICE for construction and standardization details.

Library references:

- Mehl et al., *Computational Materials Science* 136, S1-S828 (2017),
  <https://doi.org/10.1016/j.commatsci.2017.01.017>.
- Hicks et al., *Computational Materials Science* 199, 110450 (2021),
  <https://doi.org/10.1016/j.commatsci.2021.110450>.
- Eckert et al., *Computational Materials Science* 240, 112988 (2024),
  <https://doi.org/10.1016/j.commatsci.2024.112988>.

## Dependencies

ASE, spglib, NumPy, matplotlib, Pillow and PyYAML are installed separately under
their respective licenses. They are not relicensed by this repository. Model
identifiers and provider names in evaluation records are supplied by third-party
API services and do not imply endorsement or independent identity verification.
