#!/usr/bin/env bash

export WS=/data/horse/ws/chwu350f-g2i
export G2I=$WS/Gene2Image

export GMT_HALLMARK=$G2I/gmt/msigdb_2023.2_Hs/h.all.v2023.2.Hs.symbols.gmt

# Optional: only needed when INCLUDE_REACTOME=1
export GMT_REACTOME=$G2I/gmt/msigdb_2023.2_Hs/c2.cp.reactome.v2023.2.Hs.symbols.gmt
