# Attribution

The CNN + Transformer and Fourier models and the paper-method training baseline
derive from Jorge Guijarro-Ordonez and Greg Zanotti's
[released implementation](https://github.com/gregzanotti/dlsa-public), pinned at
`ea8cc2958943eb1fe914aa4fad6998994a678323`. The original copyright notice and
noncommercial redistribution terms are retained in [LICENSE](LICENSE).

Reference: Jorge Guijarro-Ordonez, Markus Pelger, and Greg Zanotti,
*Deep Learning Statistical Arbitrage*, Management Science 72(9), 7502–7549,
[doi:10.1287/mnsc.2022.03132](https://doi.org/10.1287/mnsc.2022.03132).

The frozen return panel was constructed from Yahoo Finance adjusted closing
prices; daily risk-free returns come from the Kenneth French Data Library.
The fixed stock universe was selected from November 2026 iShares holdings.
See [data/provenance.json](data/provenance.json) for the data snapshot and
selection convention. The software license does not grant rights to provider data.
