# Lebedev quadrature stencils for the Lattice-Boltzmann radiation module

Files `Lebedev<order>`: one direction per line, columns `w, cx, cy, cz, theta, phi`
(weights sum to 1; the set integrates real spherical harmonics exactly up to `l = order`).
Header: line 1 `# count`, line 2 the number of directions, line 3 the column names.

| order | directions |
|---|---|
| 5 | 14 (streaming stencil: samples the geodesic map for the spherical-harmonic fit) |
| 23 | 194 |
| 29 | 302 |
| 31 | 350 |
| 35 | 434 |
| 41 | 590 |
| 47 | 770 |
| 53 | 974 |

Copied verbatim from the reference GRLBM code
`https://github.com/Tom-Olsen/3dRadiation` (`stencils/LebedevStencil/`), MIT licence,
Copyright (c) 2024 Tom Olsen.  Generated there with `sphericalquadpy`.

If you use these in published work please cite
Olsen & Rezzolla, *General-Relativistic Lattice-Boltzmann Method for Radiation Transport*,
MNRAS (2025), arXiv:2502.17552, and the software record in that repository's `CITATION.cff`.

The table compiled into a GRACE build is selected at configure time with
`-DGRACE_LBM_STENCIL=<name>` (default `Lebedev29`); the direction count is read from line 2.
