* Sallen-Key low-pass section fed by a first-order high-pass section.
* Nominal component values; sample tolerances/faults are applied by simulate.py.
.param RHP=10k CHP=100n
.param R1V=10k R2V=10k C1V=10n C2V=10n
.param AOL=100k ROUT=100 RC2SHORT=1T
VINPUT IN 0 SINE(0 1 1k)
CHP1 IN HP {CHP}
RHP1 HP 0 {RHP}
R1 HP N1 {R1V}
R2 N1 NP {R2V}
C1 N1 0 {C1V}
C2 NP OUT {C2V}
RC2FAULT NP OUT {RC2SHORT}
RF OUT NM 5.9k
RG NM 0 10k
BOP RAW 0 V=limit({AOL}*(V(NP)-V(NM)),-12,12)
ROUT1 RAW OUT {ROUT}
CLOAD OUT 0 100p
.save V(IN) V(OUT)
