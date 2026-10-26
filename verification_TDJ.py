# -*- coding: utf-8 -*-
"""
Vérification algorithmique de TOUTES les conclusions des corrigés
(TD1, TD2, DM2, Examen, Test-QCM partie calculable).
Exécution : python3 verification_TDJ.py   -> chaque test affiche OK ou lève une erreur.
"""
from fractions import Fraction as F
import itertools
import numpy as np
from scipy.optimize import linprog
import sympy as sp

OK = 0
def check(cond, msg):
    global OK
    assert cond, "ECHEC : " + msg
    OK += 1
    print("  [OK]", msg)

# ---------------------------------------------------------------- outils
def alpha_beta(A):
    a = max(min(r) for r in A)
    b = min(max(A[i][j] for i in range(len(A))) for j in range(len(A[0])))
    return a, b

def saddle_points(A):
    m, n = len(A), len(A[0]); pts = []
    for i in range(m):
        for j in range(n):
            if A[i][j] == min(A[i]) and A[i][j] == max(A[k][j] for k in range(m)):
                pts.append((i+1, j+1))
    return pts

def solve_matrix_game(A):
    """LP : max v s.c. x^T A >= v, sum x = 1 (joueur 1) et dual (joueur 2)."""
    A = np.array(A, dtype=float); m, n = A.shape
    c = np.zeros(m+1); c[-1] = -1
    Aub = np.hstack([-A.T, np.ones((n, 1))]); bub = np.zeros(n)
    Aeq = np.hstack([np.ones((1, m)), np.zeros((1, 1))])
    r1 = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=[1],
                 bounds=[(0, None)]*m+[(None, None)])
    c2 = np.zeros(n+1); c2[-1] = 1
    Aub2 = np.hstack([A, -np.ones((m, 1))])
    Aeq2 = np.hstack([np.ones((1, n)), np.zeros((1, 1))])
    r2 = linprog(c2, A_ub=Aub2, b_ub=np.zeros(m), A_eq=Aeq2, b_eq=[1],
                 bounds=[(0, None)]*n+[(None, None)])
    return r1.x[:m], r2.x[:n], r1.x[-1], r2.x[-1]

def is_optimal(A, x, y, v):
    """Théorème 2 du cours, en arithmétique exacte."""
    m, n = len(A), len(A[0])
    col = [sum(A[i][j]*x[i] for i in range(m)) for j in range(n)]
    row = [sum(A[i][j]*y[j] for j in range(n)) for i in range(m)]
    return all(c >= v for c in col) and all(r <= v for r in row) \
        and sum(x) == 1 and sum(y) == 1 and min(x) >= 0 and min(y) >= 0

def pure_nash(P1, P2):
    m, n = len(P1), len(P1[0]); eq = []
    for i in range(m):
        for j in range(n):
            if P1[i][j] == max(P1[k][j] for k in range(m)) and \
               P2[i][j] == max(P2[i][l] for l in range(n)):
                eq.append((i+1, j+1))
    return eq

def mixed_2x2(P1, P2):
    # q = proba colonne 1 rendant J1 indifférent ; p = proba ligne 1 rendant J2 indifférent
    a, b, c, d = P1[0][0], P1[0][1], P1[1][0], P1[1][1]
    q = F(d-b, (a-b)-(c-d))
    e, f, g, h = P2[0][0], P2[0][1], P2[1][0], P2[1][1]
    p = F(h-g, (e-g)-(f-h))
    return p, q

# ======================================================================
print("\n=== TD1 (jeux matriciels) ===")
# Ex1
A = [[1, 3, 0], [1, 2, 4]]
col = lambda j: [A[i][j] for i in range(2)]
check(all(col(0)[i] < col(1)[i] for i in range(2)), "TD1-Ex1 : C1 < C2 strictement (C2 strictement dominée)")
# C3 non dominée par mixte de C1,C2 : ligne 1 donne mu*1+(1-mu)*3 >= 1 > 0
check(all(F(k, 100)*1 + (1-F(k, 100))*3 > 0 for k in range(101)), "TD1-Ex1 : C3 non dominée (ligne 1 impossible)")
check(alpha_beta(A) == (1, 1) and saddle_points(A) == [(2, 1)], "TD1-Ex1 : point selle (2,1), v=1")

# Ex2
A = [[2, 6, 0], [4, 2, 8]]
check(alpha_beta(A) == (2, 4), "TD1-Ex2 : alpha=2 < beta=4")
x, y, v = [F(1, 3), F(2, 3)], [F(2, 3), F(1, 3), 0], F(10, 3)
check(is_optimal(A, x, y, v), "TD1-Ex2 : x=(1/3,2/3), y=(2/3,1/3,0), v=10/3")
xs, ys, v1, v2 = solve_matrix_game(A)
check(abs(v1-10/3) < 1e-7 and abs(v2-10/3) < 1e-7, "TD1-Ex2 : LP scipy confirme v=10/3")

# Ex3 Von Neumann : f = y^2 - x^2
xs_, ys_ = sp.symbols('x y', real=True); f = ys_**2 - xs_**2
check(sp.diff(f, xs_, 2) == -2 and sp.diff(f, ys_, 2) == 2, "TD1-Ex3 : concave en x, convexe en y")
grid = np.linspace(-1, 1, 201)
check(all(-(t**2) <= 0 <= s**2 for t in grid for s in grid), "TD1-Ex3 : (0,0) point selle, v=0")

# Ex4
A = [[10, 8, 6, 4, 2, 3], [1, 2, 4, 3, 12, 16]]
x = [F(9, 11), F(2, 11)]; y = [0, 0, 0, F(10, 11), F(1, 11), 0]; v = F(42, 11)
check(is_optimal(A, x, y, v), "TD1-Ex4 : x=(9/11,2/11), y=(0,0,0,10/11,1/11,0), v=42/11")
xs, ys, v1, v2 = solve_matrix_game(A)
check(abs(v1-42/11) < 1e-7, "TD1-Ex4 : LP scipy v=%.6f" % v1)
p = [F(3, 14), F(1, 21)]
cons = [sum(A[i][j]*p[i] for i in range(2)) for j in range(6)]
check(cons[3] == 1 and cons[4] == 1 and all(cons[j] > 1 for j in (0, 1, 2, 5)),
      "TD1-Ex4 : P=(3/14,1/21), seules contraintes 4 et 5 saturées -> q1=q2=q3=q6=0")
check(sum(p) == F(11, 42), "TD1-Ex4 : sum P = 11/42 = 1/v")
q4, q5 = F(5, 21), F(1, 42)
check(4*q4+2*q5 == 1 and 3*q4+12*q5 == 1, "TD1-Ex4 : q4=5/21, q5=1/42")
# lignes du jeu mixte : verif enveloppe
# frontière de la région admissible de (I') : sommets et valeur de l'objectif
Acol = [(10, 1), (8, 2), (6, 4), (4, 3), (2, 12), (3, 16)]
verts = []
for (a1, b1), (a2, b2) in itertools.combinations(Acol + [(1, 0), (0, 1)], 2):
    det = a1*b2 - a2*b1
    if det == 0: continue
    r1 = 0 if (a1, b1) in [(1, 0), (0, 1)] else 1
    r2 = 0 if (a2, b2) in [(1, 0), (0, 1)] else 1
    P = (F(r1*b2 - r2*b1, det), F(a1*r2 - a2*r1, det))
    if P[0] >= 0 and P[1] >= 0 and all(a*P[0]+b*P[1] >= 1 for a, b in Acol):
        verts.append(P)
verts = sorted(set(verts))
check(verts == [(0, 1), (F(1, 13), F(3, 13)), (F(3, 14), F(1, 21)), (F(1, 2), 0)],
      "TD1-Ex4 : sommets de (I') = (0,1),(1/13,3/13),(3/14,1/21),(1/2,0)")
check(min(verts, key=sum) == (F(3, 14), F(1, 21)), "TD1-Ex4 : min de P1+P2 au sommet (3/14,1/21)")
check(F(1)+9*F(9, 11) > v and 3+F(9, 11) == v and 12-10*F(9, 11) == v, "TD1-Ex4 : intersection C4/C5 en p=9/11")

# ======================================================================
print("\n=== TD2 (équilibres de Nash) ===")
P1 = [[10, 0], [8, 7]]; P2 = [[10, 8], [0, 7]]
check(pure_nash(P1, P2) == [(1, 1), (2, 2)], "TD2-Ex1 : NE purs (C,C) et (L,L)")
p, q = mixed_2x2(P1, P2)
check(p == F(7, 9) and q == F(7, 9), "TD2-Ex1 : NE mixte p=q=7/9")
check(10*q == F(70, 9), "TD2-Ex1 : gain NE mixte 70/9")
outs = [(P1[i][j], P2[i][j]) for i in range(2) for j in range(2)]
pareto = [o for o in outs if not any(u[0] >= o[0] and u[1] >= o[1] and u != o for u in outs)]
check(pareto == [(10, 10)], "TD2-Ex1 : seul optimum de Pareto (10,10)")

P1 = [[-3, 2], [-1, 0]]; P2 = [[-3, -1], [2, 0]]
check(pure_nash(P1, P2) == [(1, 2), (2, 1)], "TD2-Ex2 : NE purs (1,2) et (2,1)")
p, q = mixed_2x2(P1, P2)
check(p == F(1, 2) and q == F(1, 2), "TD2-Ex2 : NE mixte (1/2,1/2)")
check(F(1, 2)*(-3)+F(1, 2)*2 == F(-1, 2), "TD2-Ex2 : gain NE mixte -1/2")

# Ex3 vaches
rend = {0: 0, 1: 8, 2: 5, 3: 3, 4: 2}
G1 = [[a*rend[a+b] for b in range(3)] for a in range(3)]
G2 = [[b*rend[a+b] for b in range(3)] for a in range(3)]
check(G1 == [[0, 0, 0], [8, 5, 3], [10, 6, 4]], "TD2-Ex3 : matrice J1")
check(all(G1[2][j] > G1[1][j] > G1[0][j] for j in range(3)), "TD2-Ex3 : 2 vaches strictement dominante")
check(pure_nash(G1, G2) == [(3, 3)], "TD2-Ex3 : unique NE (2,2) -> gains (4,4)")
tot = {(a, b): G1[a][b]+G2[a][b] for a in range(3) for b in range(3)}
check(max(tot.values()) == 10 and sorted(k for k, s in tot.items() if s == 10) == [(0, 2), (1, 1), (2, 0)],
      "TD2-Ex3 : optimum social total 10 en (1,1),(0,2),(2,0)")
def with_S(S):
    H1 = [[G1[a][b] - (S if a == 2 else 0) for b in range(3)] for a in range(3)]
    H2 = [[G2[a][b] - (S if b == 2 else 0) for b in range(3)] for a in range(3)]
    return pure_nash(H1, H2)
check(with_S(F(3, 2)) == [(2, 2)] and with_S(F(5, 2)) == [(2, 2)], "TD2-Ex3 : S>1 -> NE unique (1,1)")
check((3, 3) in with_S(1), "TD2-Ex3 : S=1 insuffisant : (2 vaches,2 vaches) reste NE -> il faut S>1  %s" % with_S(1))
check(all(8 > 10-F(5, 2)+0 and 5 > 6-F(5, 2) and 3 > 4-F(5, 2) for _ in [0]), "TD2-Ex3 : S>2 -> respecter le contrat strictement dominant")

# Ex4 continu
x1, x2 = sp.symbols('x1 x2', real=True)
f1 = -x1**2 + 2*x1*x2; f2 = -x2**2 + 2*x1*x2
check(sp.solve(sp.diff(f1, x1), x1) == [x2] and sp.diff(f1, x1, 2) == -2, "TD2-Ex4 : R1(x2)=x2 (concave)")
check(sp.solve(sp.diff(f2, x2), x2) == [x1], "TD2-Ex4 : R2(x1)=x1")
g = np.linspace(-1, 1, 81)
for t in g:   # chaque (t,t) est NE : aucune déviation sur une grille fine
    assert all(-s*s+2*s*t <= t*t + 1e-12 for s in np.linspace(-1, 1, 401))
check(True, "TD2-Ex4 : toute la diagonale {(t,t)} est NE (vérif. numérique)")

# Ex5 Cournot
p_ = sp.symbols('p', positive=True)
F1 = (p_ - x1 - x2)*x1
br = sp.solve(sp.diff(F1, x1), x1)[0]
check(sp.simplify(br - (p_-x2)/2) == 0, "TD2-Ex5 : R1(x2)=(p-x2)/2 si x2<p")
sol = sp.solve([sp.Eq(x1, (p_-x2)/2), sp.Eq(x2, (p_-x1)/2)], [x1, x2])
check(sol[x1] == p_/3 and sol[x2] == p_/3, "TD2-Ex5 : NE (p/3,p/3)")
check(sp.simplify(F1.subs({x1: p_/3, x2: p_/3}) - p_**2/9) == 0, "TD2-Ex5 : profit p^2/9")

# ======================================================================
print("\n=== DM2 ===")
A = [[F(1, 4), F(-1, 2)], [F(1, 6), F(-1, 2)], [F(1, 6), F(1, 6)]]
R = A
check(R[0][0] > R[1][0] and R[0][1] == R[1][1], "DM2 : L2 dominée (large) par L1")
check(R[2][0] == R[1][0] and R[2][1] > R[1][1], "DM2 : L2 dominée (large) par L3")
mix = [F(1, 2)*R[0][j]+F(1, 2)*R[2][j] for j in range(2)]
check(all(mix[j] > R[1][j] for j in range(2)), "DM2 : L2 strictement dominée par (1/2,0,1/2) : %s" % mix)
check(all(A[i][1] <= A[i][0] for i in range(3)) and A[2][1] == A[2][0], "DM2 : C1 dominée (large, non stricte) par C2")
check(alpha_beta(A) == (F(1, 6), F(1, 6)) and saddle_points(A) == [(3, 2)], "DM2 : point selle unique (3,2), v=1/6")
# ensemble des solutions optimales
v = F(1, 6)
for k in range(0, 91):
    t = F(k, 90)
    ok = is_optimal(A, [0, 0, 1], [t, 1-t], v)
    assert ok == (t <= F(8, 9)), t
check(True, "DM2 : y=(t,1-t) optimal <=> t in [0,8/9]")
for k in range(0, 21):
    for l in range(0, 21-k):
        x = [F(k, 20), F(l, 20), 1-F(k, 20)-F(l, 20)]
        colmin = min(sum(A[i][j]*x[i] for i in range(3)) for j in range(2))
        assert (colmin >= v) == (x == [0, 0, 1])
check(True, "DM2 : x*=(0,0,1) unique stratégie optimale de J1")
xs, ys, v1, _ = solve_matrix_game(A)
check(abs(v1-1/6) < 1e-8, "DM2 : LP scipy v=1/6")

# DM2 Ex2 Partie 2
f1 = x1*x2 - x1**2*x2**2; f2 = x1 + x2 - (x1**2 + x2**2)
check(sp.solve(sp.diff(f1, x1), x1) == [1/(2*x2)], "DM2-P2 : R1(x2)=1/(2x2), x2!=0")
check(sp.diff(f1, x1, 2) == -2*x2**2, "DM2-P2 : f1 concave en x1 (strict si x2!=0)")
check(sp.solve(sp.diff(f2, x2), x2) == [sp.Rational(1, 2)], "DM2-P2 : R2(x1)=1/2")
check(f1.subs({x1: 1, x2: sp.Rational(1, 2)}) == sp.Rational(1, 4) and
      f2.subs({x1: 1, x2: sp.Rational(1, 2)}) == sp.Rational(1, 4), "DM2-P2 : NE (1,1/2), gains (1/4,1/4)")
check(sp.expand(f2.subs(x2, sp.Rational(1, 2)) - f2 - (x2-sp.Rational(1, 2))**2) == 0,
      "DM2-P2 : f2(x1,1/2)-f2(x1,x2)=(x2-1/2)^2 -> 1/2 strictement dominante")
check(sp.solve(sp.diff(f1.subs(x2, 1), x1), x1) == [sp.Rational(1, 2)], "DM2-P2 : R1(1)=1/2 != R1(1/2)=1 -> pas de dominante pour J1")

# ======================================================================
print("\n=== EXAMEN ===")
A = [[1, 3, 0], [2, 1, 4]]
check(alpha_beta(A) == (1, 2), "EX-Ex1 J : alpha=1<beta=2")
check(is_optimal(A, [F(1, 3), F(2, 3)], [F(2, 3), F(1, 3), 0], F(5, 3)), "EX-Ex1 J : v=5/3")
B = [[2, 6, 5, 0], [4, 2, 3, 8], [4, 2, 2, 6], [2, 4, 4, 2]]
check(is_optimal(B, [F(1, 3), F(2, 3), 0, 0], [F(2, 3), F(1, 3), 0, 0], F(10, 3)), "EX-Ex1 J' : v=10/3")
mix = [F(2, 3)*B[0][j]+F(1, 3)*B[1][j] for j in range(4)]
check(all(mix[j] > B[3][j] for j in range(4)), "EX-Ex1 J' : L4 < (2/3)L1+(1/3)L2")
mixc = [F(1, 3)*B[i][0]+F(2, 3)*B[i][1] for i in range(2)]
check(all(mixc[i] < B[i][2] for i in range(2)), "EX-Ex1 J' : C3 > (1/3)C1+(2/3)C2")
xs, ys, v1, _ = solve_matrix_game(B)
check(abs(v1-10/3) < 1e-7, "EX-Ex1 J' : LP scipy v=10/3")

print("\n=== TEST (QCM, items calculables) ===")
A = [[5, 3, 4, 3], [7, 2, 0, -2], [0, -1, -4, 2]]   # exemple cours p.8
check(saddle_points(A) == [(1, 2), (1, 4)], "Cours p.8 : points selle (1,2),(1,4), v=3")


# ======================================================================
print("\n=== GUIDE : exemples du cours et exercices d'entraînement ===")
# Cours Chap.2 p.6-7 : bimatriciel
P1 = [[8, 4, 10], [9, 5, 6]]; P2 = [[8, 7, 6], [2, 5, 3]]
check(pure_nash(P1, P2) == [(2, 2)], "Cours Ch2 p6 : unique NE (t2,s2)")
# Cours Ch1 p19 : après éliminations, jeu [[3,-2],[-1,3]]
A = [[1, 2, 3, -7], [3, 8, 6, -2], [-1, 2, 7, 3]]
check(all(A[0][j] < A[1][j] for j in range(4)), "Cours Ch1 p19 : x1 strictement dominée par x2")
A2 = [[3, -2], [-1, 3]]
check(saddle_points(A2) == [], "Cours Ch1 p20 : [[3,-2],[-1,3]] sans point selle")
def f2x2(M):
    a, b = M[0]; c, d = M[1]; D = a+d-b-c
    return F(d-c, D), F(d-b, D), F(a*d-b*c, D)
x1, y1, v = f2x2(A2)
check(is_optimal(A2, [x1, 1-x1], [y1, 1-y1], v) and v == F(7, 9), "Cours Ch1 p20 (complément) : x=(4/9,5/9), y=(5/9,4/9), v=7/9")
check(is_optimal(A, [0, x1, 1-x1], [y1, 0, 0, 1-y1], v), "Cours Ch1 p19 : solution remontée au jeu 3x4, v=7/9")
# Cours Ch1 p20 : 4x4 de valeur 1
A = [[1, -1, 2, 2], [1, 3, -2, 1], [0, -1, 3, 1], [1, 2, 1, 3]]
check(alpha_beta(A) == (1, 1), "Cours Ch1 p20 : alpha=beta=1")
# Forme extensive MAC/CAM (induction à rebours)
pay = {('MAC', 'MAC'): (200, 600), ('MAC', 'CAM'): (100, 100), ('CAM', 'MAC'): (100, 100), ('CAM', 'CAM'): (600, 200)}
br2 = {a: max(['MAC', 'CAM'], key=lambda b: pay[(a, b)][1]) for a in ['MAC', 'CAM']}
a1 = max(['MAC', 'CAM'], key=lambda a: pay[(a, br2[a])][0])
check(br2 == {'MAC': 'MAC', 'CAM': 'CAM'} and a1 == 'CAM' and pay[(a1, br2[a1])] == (600, 200),
      "Cours Ch1 p33 (complément) : induction à rebours -> (CAM,CAM), gains (600,200)")
# Entraînement E1 : formule 2x2
A = [[3, 1], [0, 2]]
x1, y1, v = f2x2(A)
check(saddle_points(A) == [] and (x1, y1, v) == (F(1, 2), F(1, 4), F(3, 2)) and is_optimal(A, [x1, 1-x1], [y1, 1-y1], v),
      "Entraînement E1 : x=(1/2,1/2), y=(1/4,3/4), v=3/2")
# E2 dilemme du prisonnier
check(pure_nash([[3, 0], [5, 1]], [[3, 5], [0, 1]]) == [(2, 2)], "Entraînement E2 : NE unique (D,D)")
# E3 jeu continu f = y^2 - x^2 + 2x - 2y sur [0,2]^2
xx, yy = sp.symbols('x y', real=True); f = yy**2 - xx**2 + 2*xx - 2*yy
check(sp.expand(f.subs(yy, 1) + (xx-1)**2) == 0 and sp.expand(f.subs(xx, 1) - (yy-1)**2) == 0, "Entraînement E3 : point selle (1,1), v=0")
# E4 graphique 2x3
A = [[4, 1, 3], [0, 3, 2]]
check(alpha_beta(A) == (1, 3), "Entraînement E4 : alpha=1, beta=3")
check(is_optimal(A, [F(1, 2), F(1, 2)], [F(1, 3), F(2, 3), 0], 2), "Entraînement E4 : x=(1/2,1/2), y=(1/3,2/3,0), v=2")
check(all(F(1, 2)*A[i][0]+F(1, 2)*A[i][1] < A[i][2] for i in range(2)), "Entraînement E4 : C3 strictement dominée par (1/2,1/2,0)")
# Invariance affine : translation k pour rendre v>0 (DM2 : A+1)
A = [[F(1, 4), F(-1, 2)], [F(1, 6), F(-1, 2)], [F(1, 6), F(1, 6)]]
B = [[a+1 for a in r] for r in A]
check(is_optimal(B, [0, 0, 1], [0, 1], F(7, 6)), "Invariance : A+1 -> même solution, v+1=7/6")

print("\nTOTAL : %d vérifications réussies." % OK)
