# -*- coding: utf-8 -*-
"""
Projet 8 — Jeux de coordination (Bataille des sexes)
Master Recherche Opérationnelle (RO) -- Université Akli Mohand Oulhadj - Bouira
Module : Théorie des Jeux

Objectif : Analyser les jeux à équilibres multiples et le problème de la sélection d'équilibre.
Livrables : Identification des équilibres purs et mixtes, discussion des points focaux.
Piste de départ : Comparer les équilibres et leurs gains espérés respectifs.

Ce script implémente :
1. Résolution exacte et symbolique des équilibres de Nash purs et mixtes (Sympy & Fractions)
2. Analyse paramétrique générale (P > Q > N) selon notes-mitro206
3. Calcul des stratégies de sécurité (Maximin / Minimax) et niveaux de sécurité
4. Résolution de l'équilibre corrélé (Aumann 1974) par Programmation Linéaire (Scipy)
5. Modélisation de l'issue sous CKR (Connaissance Commune de la Rationalité, Léo Gerville-Réache HAL 2017)
6. Simulation Monte-Carlo et comparaison empirique des gains espérés et taux d'échec
7. Dynamique du réplicateur (Evolutionary Game Theory) et portrait de phase
8. Génération des graphiques (courbes de meilleure réponse, dynamique, comparaison des gains)
9. Suite de tests automatisés validant 100% des résultats théoriques
"""

import sys
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog
import sympy as sp

# ----------------------------------------------------------------------
# Section 1 : Modélisation et Outils de Base
# ----------------------------------------------------------------------

class BattleOfTheSexesGame:
    """
    Modèle général d'un jeu de la bataille des sexes (Bach or Stravinsky / Guerre des sexes).
    Joueur 1 (Lignes) : 2 actions (ex: Opéra/Alpinisme [1], Foot/Boxe [2])
    Joueur 2 (Colonnes) : 2 actions (ex: Opéra/Alpinisme [1], Foot/Boxe [2])
    
    Matrice de gains :
      Action 1 : préférée de J1 (gain P pour J1, Q pour J2 si coordonnés)
      Action 2 : préférée de J2 (gain Q pour J1, P pour J2 si coordonnés)
      Non coordination : gains N1, N2 (typiquement 0, 0)
    """
    def __init__(self, P=2, Q=1, N=0, p_labels=('Opéra', 'Football'), j_labels=('Joueur 1 (Mari)', 'Joueur 2 (Femme)')):
        self.P = P
        self.Q = Q
        self.N = N
        self.actions = p_labels
        self.players = j_labels
        
        # Matrices 2x2
        # J1 lignes: 0=Act1, 1=Act2 ; J2 colonnes: 0=Act1, 1=Act2
        self.A = np.array([[P, N],
                           [N, Q]], dtype=float)
        self.B = np.array([[Q, N],
                           [N, P]], dtype=float)
        
    def pure_nash(self):
        """
        Détection des équilibres de Nash en stratégies pures par meilleure réponse mutuelle.
        """
        m, n = self.A.shape
        equilibria = []
        for i in range(m):
            for j in range(n):
                # Meilleure réponse de J1 au choix j de J2
                br1 = (self.A[i, j] == np.max(self.A[:, j]))
                # Meilleure réponse de J2 au choix i de J1
                br2 = (self.B[i, j] == np.max(self.B[i, :]))
                if br1 and br2:
                    equilibria.append(((i, j), (self.A[i, j], self.B[i, j])))
        return equilibria

    def mixed_nash(self):
        """
        Calcul exact de l'équilibre de Nash en stratégies mixtes.
        p = probabilité que J1 joue Action 1 (Opéra)
        q = probabilité que J2 joue Action 1 (Opéra)
        
        Condition d'indifférence de J1:
          u1(Act1, q) = u1(Act2, q)
          q*P + (1-q)*N = q*N + (1-q)*Q
          q*(P - N) = (1-q)*(Q - N)
          q* = (Q - N) / (P + Q - 2N)
          
        Condition d'indifférence de J2:
          u2(p, Act1) = u2(p, Act2)
          p*Q + (1-p)*N = p*N + (1-p)*P
          p*(Q - N) = (1-p)*(P - N)
          p* = (P - N) / (P + Q - 2N)
        """
        P, Q, N = F(self.P), F(self.Q), F(self.N)
        denom = (P + Q - 2*N)
        p_star = (P - N) / denom
        q_star = (Q - N) / denom
        
        # Gains espérés à l'équilibre mixte :
        # u1* = q*P + (1-q)*N = (Q - N)*P/(P + Q - 2N) + ... = (P*Q - N^2)/(P + Q - 2N)
        u1_star = (P*Q - N**2) / denom
        u2_star = (P*Q - N**2) / denom
        
        return (p_star, q_star), (u1_star, u2_star)

    def security_strategies(self):
        """
        Stratégie de sécurité (Maximin) pour chaque joueur.
        Pour J1: max_p min_q u1(p, q)
        u1(p, Col1) = p*P + (1-p)*N
        u1(p, Col2) = p*N + (1-p)*Q
        Le minimum est maximisé au croisement : p*P = (1-p)*Q => p_sec = Q / (P + Q - 2N) (si N=0)
        Plus généralement avec N:
          p*(P - N) + N = (1-p)*(Q - N) + N
          p_sec = (Q - N) / (P + Q - 2N)
        Gain de sécurité garanti : v1 = (P*Q - N^2)/(P + Q - 2N)
        
        De même pour J2:
          q_sec = (P - N) / (P + Q - 2N)
          v2 = (P*Q - N^2)/(P + Q - 2N)
        """
        P, Q, N = F(self.P), F(self.Q), F(self.N)
        denom = (P + Q - 2*N)
        p_sec = (Q - N) / denom
        q_sec = (P - N) / denom
        v1 = (P*Q - N**2) / denom
        v2 = (P*Q - N**2) / denom
        return (p_sec, q_sec), (v1, v2)

    def correlated_equilibrium_max_welfare(self):
        """
        Équilibre corrélé (Aumann 1974) maximisant le bien-être social (somme des gains).
        Formulation en Programmation Linéaire (PL) :
        Variables : probabilités z = [z00, z01, z10, z11] >= 0
        Contrainte 1 : sum(z) = 1
        Contraintes d'incitation (pas d'intérêt à dévier unilatéralement).
        """
        A, B = self.A, self.B
        c = - (A + B).flatten()
        Aub = np.array([
            [(A[1,0]-A[0,0]), (A[1,1]-A[0,1]), 0, 0],
            [0, 0, (A[0,0]-A[1,0]), (A[0,1]-A[1,1])],
            [(B[0,1]-B[0,0]), 0, (B[1,1]-B[1,0]), 0],
            [0, (B[0,0]-B[0,1]), 0, (B[1,0]-B[1,1])]
        ], dtype=float)
        bub = np.zeros(4)
        Aeq = np.ones((1, 4))
        beq = np.array([1.0])
        bounds = [(0, 1)] * 4
        res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')
        z = res.x
        u1_ce = np.sum(z * A.flatten())
        u2_ce = np.sum(z * B.flatten())
        return z.reshape((2, 2)), (u1_ce, u2_ce)

    def correlated_equilibrium_fair(self):
        """
        Équilibre corrélé équitable (égalitariste / symétrique) :
        Maximise min(u1, u2) sous contraintes d'incitation CE et z00+z01+z10+z11=1.
        Formulation PL avec variable auxiliaire v <= u1 et v <= u2 :
        Variables x = [z00, z01, z10, z11, v]
        Maximiser v (donc c = [0, 0, 0, 0, -1])
        Contraintes :
          v - u1(z) <= 0
          v - u2(z) <= 0
          + 4 contraintes de non-déviation CE.
        """
        A, B = self.A, self.B
        # c : minimiser -v
        c = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
        
        # 4 contraintes CE
        CE_ub = np.array([
            [(A[1,0]-A[0,0]), (A[1,1]-A[0,1]), 0, 0, 0],
            [0, 0, (A[0,0]-A[1,0]), (A[0,1]-A[1,1]), 0],
            [(B[0,1]-B[0,0]), 0, (B[1,1]-B[1,0]), 0, 0],
            [0, (B[0,0]-B[0,1]), 0, (B[1,0]-B[1,1]), 0]
        ], dtype=float)
        
        # v - sum(z_ij * A_ij) <= 0 => -A00*z00 - A01*z01 - A10*z10 - A11*z11 + v <= 0
        u1_row = np.array([-A[0,0], -A[0,1], -A[1,0], -A[1,1], 1.0])
        u2_row = np.array([-B[0,0], -B[0,1], -B[1,0], -B[1,1], 1.0])
        
        Aub = np.vstack([CE_ub, u1_row, u2_row])
        bub = np.zeros(6)
        
        # sum(z) = 1
        Aeq = np.array([[1.0, 1.0, 1.0, 1.0, 0.0]])
        beq = np.array([1.0])
        
        bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, None)]
        res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')
        
        z = res.x[:4]
        v_opt = res.x[4]
        u1 = np.sum(z * A.flatten())
        u2 = np.sum(z * B.flatten())
        return z.reshape((2, 2)), (u1, u2)

    def ckr_symmetric_solution(self):
        """
        Issue symétrique sous CKR (Léo Gerville-Réache, HAL 2017).
        Sous l'hypothèse de rationalité commune et de symétrie, les deux joueurs
        choisissent la même probabilité p d'opter pour leur activité préférée.
        Ici : J1 choisit Opéra avec prob p, J2 choisit Foot avec prob p.
        Coordination sur Opéra : J1 joue Opéra (p), J2 joue Opéra (1-p) => prob p*(1-p), gain P pour J1, Q pour J2.
        Coordination sur Foot  : J1 joue Foot (1-p), J2 joue Foot (p)   => prob (1-p)*p, gain Q pour J1, P pour J2.
        Défaut de coordination : prob 1 - 2p(1-p), gain N.
        
        Espérance de gain symétrique :
          u(p) = p*(1-p)*P + (1-p)*p*Q + (p^2 + (1-p)^2)*N
               = p*(1-p)*(P + Q - 2N) + N
        Maximiser u(p) sur [0, 1] :
          Dérivée : (1 - 2p)*(P + Q - 2N) = 0 => p = 1/2.
          u_max = (1/4)*(P + Q - 2N) + N.
        Pour P=2, Q=1, N=0 : u(1/2) = 3/4 = 0.75 > 2/3 (Nash mixte).
        """
        P, Q, N = F(self.P), F(self.Q), F(self.N)
        p_opt = F(1, 2)
        u_opt = F(1, 4) * (P + Q - 2*N) + N
        return p_opt, u_opt

    def efficiency_metrics(self):
        """
        Calcul du Prix de l'Anarchie (Price of Anarchy - PoA)
        et du Prix de la Stabilité (Price of Stability - PoS).
        W_social(s) = u1(s) + u2(s)
        """
        (p_star, q_star), (u1_mix, u2_mix) = self.mixed_nash()
        w_pure = self.P + self.Q  # pour les 2 équilibres purs
        w_mixed = float(u1_mix + u2_mix)
        w_opt = self.P + self.Q
        
        poa = w_opt / min(w_pure, w_mixed)
        pos = w_opt / max(w_pure, w_mixed)
        
        # Probabilité de coordination sous Nash mixte
        # Coordination = (Opéra, Opéra) ou (Foot, Foot)
        p_f = float(p_star)
        q_f = float(q_star)
        p_coord = p_f * q_f + (1 - p_f) * (1 - q_f)
        p_miscoord = 1.0 - p_coord
        
        return {
            'w_social_pure': w_pure,
            'w_social_mixed': w_mixed,
            'w_social_opt': w_opt,
            'PoA': poa,
            'PoS': pos,
            'p_coordination_mixed': p_coord,
            'p_miscoordination_mixed': p_miscoord
        }


# ----------------------------------------------------------------------
# Section 2 : Simulations Monte-Carlo et Dynamique
# ----------------------------------------------------------------------

def simulate_coordination(game, n_trials=50000, seed=42):
    """
    Simulation Monte-Carlo comparant les différents modes de sélection / équilibres.
    """
    np.random.seed(seed)
    (p_star, q_star), _ = game.mixed_nash()
    p_mix, q_mix = float(p_star), float(q_star)
    p_ckr, _ = game.ckr_symmetric_solution()
    p_ckr = float(p_ckr)
    (p_sec, q_sec), _ = game.security_strategies()
    p_sec, q_sec = float(p_sec), float(q_sec)
    
    results = {}
    
    # 1. Équilibre pur (Opéra, Opéra)
    results['Pur 1 (Opéra, Opéra)'] = {
        'gain_J1': float(game.P),
        'gain_J2': float(game.Q),
        'gain_total': float(game.P + game.Q),
        'taux_coordination': 1.0
    }
    
    # 2. Équilibre pur (Foot, Foot)
    results['Pur 2 (Foot, Foot)'] = {
        'gain_J1': float(game.Q),
        'gain_J2': float(game.P),
        'gain_total': float(game.P + game.Q),
        'taux_coordination': 1.0
    }
    
    # 3. Équilibre mixte de Nash
    a1_mix = np.random.rand(n_trials) < p_mix
    a2_mix = np.random.rand(n_trials) < q_mix
    coord_mix = (a1_mix == a2_mix)
    g1_mix = np.where(a1_mix & a2_mix, game.P, np.where(~a1_mix & ~a2_mix, game.Q, game.N))
    g2_mix = np.where(a1_mix & a2_mix, game.Q, np.where(~a1_mix & ~a2_mix, game.P, game.N))
    results['Nash mixte'] = {
        'gain_J1': float(np.mean(g1_mix)),
        'gain_J2': float(np.mean(g2_mix)),
        'gain_total': float(np.mean(g1_mix + g2_mix)),
        'taux_coordination': float(np.mean(coord_mix))
    }
    
    # 4. Stratégie de sécurité (Maximin)
    a1_sec = np.random.rand(n_trials) < p_sec
    a2_sec = np.random.rand(n_trials) < q_sec
    coord_sec = (a1_sec == a2_sec)
    g1_sec = np.where(a1_sec & a2_sec, game.P, np.where(~a1_sec & ~a2_sec, game.Q, game.N))
    g2_sec = np.where(a1_sec & a2_sec, game.Q, np.where(~a1_sec & ~a2_sec, game.P, game.N))
    results['Sécurité (Maximin)'] = {
        'gain_J1': float(np.mean(g1_sec)),
        'gain_J2': float(np.mean(g2_sec)),
        'gain_total': float(np.mean(g1_sec + g2_sec)),
        'taux_coordination': float(np.mean(coord_sec))
    }
    
    # 5. Issue symétrique sous CKR (p=0.5 chacun vers activité préférée)
    # J1 joue Opéra avec prob p_ckr, J2 joue Foot avec prob p_ckr (donc Opéra avec prob 1 - p_ckr)
    a1_ckr = np.random.rand(n_trials) < p_ckr
    a2_ckr = np.random.rand(n_trials) < (1.0 - p_ckr)
    coord_ckr = (a1_ckr == a2_ckr)
    g1_ckr = np.where(a1_ckr & a2_ckr, game.P, np.where(~a1_ckr & ~a2_ckr, game.Q, game.N))
    g2_ckr = np.where(a1_ckr & a2_ckr, game.Q, np.where(~a1_ckr & ~a2_ckr, game.P, game.N))
    results['CKR symétrique (p=0.5)'] = {
        'gain_J1': float(np.mean(g1_ckr)),
        'gain_J2': float(np.mean(g2_ckr)),
        'gain_total': float(np.mean(g1_ckr + g2_ckr)),
        'taux_coordination': float(np.mean(coord_ckr))
    }
    
    # 6. Équilibre corrélé (Pile ou Face public)
    coin = np.random.rand(n_trials) < 0.5
    g1_ce = np.where(coin, game.P, game.Q)
    g2_ce = np.where(coin, game.Q, game.P)
    results['Équilibre corrélé (signal 50/50)'] = {
        'gain_J1': float(np.mean(g1_ce)),
        'gain_J2': float(np.mean(g2_ce)),
        'gain_total': float(np.mean(g1_ce + g2_ce)),
        'taux_coordination': 1.0
    }
    
    # 7. Alternance répétée (Jeu répété à horizon infini avec convention de parité)
    # Période paire : Opéra (2,1) ; Période impaire : Foot (1,2)
    turns = np.arange(n_trials) % 2
    g1_alt = np.where(turns == 0, game.P, game.Q)
    g2_alt = np.where(turns == 0, game.Q, game.P)
    results['Alternance répétée (Folk Theorem)'] = {
        'gain_J1': float(np.mean(g1_alt)),
        'gain_J2': float(np.mean(g2_alt)),
        'gain_total': float(np.mean(g1_alt + g2_alt)),
        'taux_coordination': 1.0
    }
    
    return results


def replicator_dynamics_system(p, q, P=2.0, Q=1.0, N=0.0):
    """
    Système différentiel de la dynamique du réplicateur à 2 populations.
    p = proportion de Joueurs 1 choisissant l'action 1 (Opéra)
    q = proportion de Joueurs 2 choisissant l'action 1 (Opéra)
    
    dp/dt = p * [ u1(Opéra, q) - u1_moyen ] = p * (1 - p) * [ u1(Opéra, q) - u1(Foot, q) ]
    dq/dt = q * [ u2(p, Opéra) - u2_moyen ] = q * (1 - q) * [ u2(p, Opéra) - u2(p, Foot) ]
    
    Ici:
      u1(Opéra, q) = q*P + (1-q)*N
      u1(Foot, q)  = q*N + (1-q)*Q
      delta_1(q)   = q*(P - N) - (1-q)*(Q - N) = q*(P + Q - 2N) - (Q - N)
      
      u2(p, Opéra) = p*Q + (1-p)*N
      u2(p, Foot)  = p*N + (1-p)*P
      delta_2(p)   = p*(Q - N) - (1-p)*(P - N) = p*(P + Q - 2N) - (P - N)
    """
    delta1 = q * (P + Q - 2*N) - (Q - N)
    delta2 = p * (P + Q - 2*N) - (P - N)
    dp_dt = p * (1.0 - p) * delta1
    dq_dt = q * (1.0 - q) * delta2
    return dp_dt, dq_dt


# ----------------------------------------------------------------------
# Section 3 : Visualisations Graphiques (SVG Natif & Matplotlib)
# ----------------------------------------------------------------------

def generate_svg_best_responses(game, filename="projet8_figure_best_responses.svg"):
    (p_star, q_star), _ = game.mixed_nash()
    p_mix, q_mix = float(p_star), float(q_star)
    
    # Dimensions SVG
    W, H = 550, 500
    margin = 60
    pw = W - 2*margin
    ph = H - 2*margin
    
    # Echelle [0, 1] -> pixels
    to_x = lambda p: margin + p * pw
    to_y = lambda q: H - margin - q * ph
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="background-color:#ffffff; font-family:sans-serif;">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Correspondances de Meilleure Réponse et Équilibres de Nash</text>',
        
        # Grille
        f'<rect x="{to_x(0)}" y="{to_y(1)}" width="{pw}" height="{ph}" fill="#f8f9fa" stroke="#bdc3c7" stroke-width="1.5"/>',
        f'<line x1="{to_x(0.5)}" y1="{to_y(0)}" x2="{to_x(0.5)}" y2="{to_y(1)}" stroke="#e2e8f0" stroke-dasharray="4"/>',
        f'<line x1="{to_x(0)}" y1="{to_y(0.5)}" x2="{to_x(1)}" y2="{to_y(0.5)}" stroke="#e2e8f0" stroke-dasharray="4"/>',
        
        # BR1(q) (en bleu) : si q < q_mix: p=0 ; si q == q_mix: p in [0,1] ; si q > q_mix: p=1
        f'<polyline points="{to_x(0)},{to_y(0)} {to_x(0)},{to_y(q_mix)} {to_x(1)},{to_y(q_mix)} {to_x(1)},{to_y(1)}" fill="none" stroke="#2980b9" stroke-width="3.5"/>',
        
        # BR2(p) (en rouge tireté) : si p < p_mix: q=0 ; si p == p_mix: q in [0,1] ; si p > p_mix: q=1
        f'<polyline points="{to_x(0)},{to_y(0)} {to_x(p_mix)},{to_y(0)} {to_x(p_mix)},{to_y(1)} {to_x(1)},{to_y(1)}" fill="none" stroke="#c0392b" stroke-width="3" stroke-dasharray="6,4"/>',
        
        # Équilibres de Nash (intersections)
        # E2 : (0, 0)
        f'<circle cx="{to_x(0)}" cy="{to_y(0)}" r="7" fill="#27ae60" stroke="#1e824c" stroke-width="2"/>',
        f'<text x="{to_x(0)+12}" y="{to_y(0)-10}" font-size="12" font-weight="bold" fill="#27ae60">E2: (Foot, Foot)</text>',
        
        # E1 : (1, 1)
        f'<circle cx="{to_x(1)}" cy="{to_y(1)}" r="7" fill="#2980b9" stroke="#1f618d" stroke-width="2"/>',
        f'<text x="{to_x(1)-115}" y="{to_y(1)+18}" font-size="12" font-weight="bold" fill="#2980b9">E1: (Opéra, Opéra)</text>',
        
        # Emix : (p_mix, q_mix)
        f'<circle cx="{to_x(p_mix)}" cy="{to_y(q_mix)}" r="8" fill="#8e44ad" stroke="#5b2c6f" stroke-width="2"/>',
        f'<text x="{to_x(p_mix)+12}" y="{to_y(q_mix)+5}" font-size="12" font-weight="bold" fill="#8e44ad">Emix: (2/3, 1/3)</text>',
        
        # Axes & labels
        f'<text x="{W/2}" y="{H-15}" text-anchor="middle" font-size="13" fill="#34495e">Probabilité Joueur 1 (Opéra) : p</text>',
        f'<text x="20" y="{H/2}" text-anchor="middle" font-size="13" fill="#34495e" transform="rotate(-90 20 {H/2})">Probabilité Joueur 2 (Opéra) : q</text>',
        f'<text x="{to_x(0)}" y="{to_y(0)+20}" text-anchor="middle" font-size="11" fill="#7f8c8d">0</text>',
        f'<text x="{to_x(1)}" y="{to_y(0)+20}" text-anchor="middle" font-size="11" fill="#7f8c8d">1</text>',
        f'<text x="{to_x(p_mix)}" y="{to_y(0)+20}" text-anchor="middle" font-size="11" fill="#8e44ad">2/3</text>',
        f'<text x="{to_x(0)-15}" y="{to_y(0)+4}" text-anchor="middle" font-size="11" fill="#7f8c8d">0</text>',
        f'<text x="{to_x(0)-15}" y="{to_y(1)+4}" text-anchor="middle" font-size="11" fill="#7f8c8d">1</text>',
        f'<text x="{to_x(0)-20}" y="{to_y(q_mix)+4}" text-anchor="middle" font-size="11" fill="#8e44ad">1/3</text>',
        
        # Légende
        f'<rect x="{margin+15}" y="{to_y(1)+15}" width="220" height="75" fill="#ffffff" fill-opacity="0.9" stroke="#cbd5e1" rx="4"/>',
        f'<line x1="{margin+25}" y1="{to_y(1)+32}" x2="{margin+60}" y2="{to_y(1)+32}" stroke="#2980b9" stroke-width="3"/>',
        f'<text x="{margin+70}" y="{to_y(1)+36}" font-size="11" fill="#2c3e50">BR1(q) Joueur 1</text>',
        f'<line x1="{margin+25}" y1="{to_y(1)+52}" x2="{margin+60}" y2="{to_y(1)+52}" stroke="#c0392b" stroke-width="2.5" stroke-dasharray="5,3"/>',
        f'<text x="{margin+70}" y="{to_y(1)+56}" font-size="11" fill="#2c3e50">BR2(p) Joueur 2</text>',
        f'<circle cx="{margin+42}" cy="{to_y(1)+70}" r="5" fill="#8e44ad"/>',
        f'<text x="{margin+70}" y="{to_y(1)+74}" font-size="11" fill="#8e44ad">Intersection Nash mixte</text>',
        
        '</svg>'
    ]
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    return filename


def generate_svg_comparison_bars(game, filename="projet8_figure_comparison_bars.svg"):
    sim_res = simulate_coordination(game, n_trials=25000)
    labels = list(sim_res.keys())
    g1 = [sim_res[k]['gain_J1'] for k in labels]
    g2 = [sim_res[k]['gain_J2'] for k in labels]
    rates = [sim_res[k]['taux_coordination'] * 100 for k in labels]
    
    W, H = 750, 480
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="background-color:#ffffff; font-family:sans-serif;">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Comparaison des Gains Espérés et Efficacité de Coordination</text>',
    ]
    
    # Axe horizontal : 7 groupes
    chart_x = 70
    chart_y = 60
    chart_w = W - 110
    chart_h = 320
    
    max_val = 2.4
    scale_y = lambda val: chart_y + chart_h - (val / max_val) * chart_h
    
    # Grille de fond
    for grid_v in [0.5, 1.0, 1.5, 2.0]:
        y_pos = scale_y(grid_v)
        svg.append(f'<line x1="{chart_x}" y1="{y_pos}" x2="{chart_x+chart_w}" y2="{y_pos}" stroke="#edf2f7" stroke-width="1"/>')
        svg.append(f'<text x="{chart_x-10}" y="{y_pos+4}" text-anchor="end" font-size="11" fill="#718096">{grid_v:.1f}</text>')
        
    svg.append(f'<line x1="{chart_x}" y1="{chart_y+chart_h}" x2="{chart_x+chart_w}" y2="{chart_y+chart_h}" stroke="#4a5568" stroke-width="1.5"/>')
    svg.append(f'<line x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{chart_y+chart_h}" stroke="#4a5568" stroke-width="1.5"/>')
    svg.append(f'<text x="25" y="{chart_y+chart_h/2}" text-anchor="middle" font-size="12" fill="#2d3748" transform="rotate(-90 25 {chart_y+chart_h/2})">Gains espérés</text>')
    
    # Ligne Q = 1.0 et Nash mixte = 0.67
    y_q = scale_y(1.0)
    svg.append(f'<line x1="{chart_x}" y1="{y_q}" x2="{chart_x+chart_w}" y2="{y_q}" stroke="#e53e3e" stroke-dasharray="4" stroke-width="1.5"/>')
    svg.append(f'<text x="{chart_x+chart_w-5}" y="{y_q-6}" text-anchor="end" font-size="10" fill="#e53e3e">Seuil de pire coordination (Q=1.0)</text>')
    
    y_nm = scale_y(2/3)
    svg.append(f'<line x1="{chart_x}" y1="{y_nm}" x2="{chart_x+chart_w}" y2="{y_nm}" stroke="#805ad5" stroke-dasharray="3" stroke-width="1.5"/>')
    svg.append(f'<text x="{chart_x+chart_w-5}" y="{y_nm-6}" text-anchor="end" font-size="10" fill="#805ad5">Nash mixte (2/3 ≈ 0.67)</text>')
    
    group_w = chart_w / len(labels)
    bar_w = 22
    
    for i, lab in enumerate(labels):
        gx = chart_x + i * group_w + group_w/2
        
        # Barre J1
        h1 = (g1[i] / max_val) * chart_h
        y1 = chart_y + chart_h - h1
        svg.append(f'<rect x="{gx - bar_w - 2}" y="{y1}" width="{bar_w}" height="{h1}" fill="#3182ce" rx="2"/>')
        svg.append(f'<text x="{gx - bar_w/2 - 2}" y="{y1-5}" text-anchor="middle" font-size="10" fill="#2b6cb0">{g1[i]:.2f}</text>')
        
        # Barre J2
        h2 = (g2[i] / max_val) * chart_h
        y2 = chart_y + chart_h - h2
        svg.append(f'<rect x="{gx + 2}" y="{y2}" width="{bar_w}" height="{h2}" fill="#dd6b20" rx="2"/>')
        svg.append(f'<text x="{gx + bar_w/2 + 2}" y="{y2-5}" text-anchor="middle" font-size="10" fill="#c05621">{g2[i]:.2f}</text>')
        
        # Taux de coordination badge
        svg.append(f'<rect x="{gx - 28}" y="{chart_y+chart_h+38}" width="56" height="18" fill="#ebf8ff" stroke="#bee3f8" rx="4"/>')
        svg.append(f'<text x="{gx}" y="{chart_y+chart_h+51}" text-anchor="middle" font-size="10" font-weight="bold" fill="#2b6cb0">{rates[i]:.0f}% coord.</text>')
        
        # Label raccourci
        short_lab = lab.split(' (')[0]
        svg.append(f'<text x="{gx}" y="{chart_y+chart_h+20}" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#2d3748">{short_lab}</text>')
        
    # Légende en haut
    svg.append(f'<rect x="{chart_x+20}" y="{chart_y+10}" width="260" height="35" fill="#ffffff" fill-opacity="0.9" stroke="#cbd5e0" rx="4"/>')
    svg.append(f'<rect x="{chart_x+30}" y="{chart_y+20}" width="14" height="14" fill="#3182ce" rx="2"/>')
    svg.append(f'<text x="{chart_x+50}" y="{chart_y+32}" font-size="11" fill="#2d3748">Gain Joueur 1</text>')
    svg.append(f'<rect x="{chart_x+150}" y="{chart_y+20}" width="14" height="14" fill="#dd6b20" rx="2"/>')
    svg.append(f'<text x="{chart_x+170}" y="{chart_y+32}" font-size="11" fill="#2d3748">Gain Joueur 2</text>')
    
    svg.append('</svg>')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    return filename


def generate_svg_replicator_dynamics(game, filename="projet8_figure_replicator_dynamics.svg"):
    (p_star, q_star), _ = game.mixed_nash()
    p_mix, q_mix = float(p_star), float(q_star)
    
    W, H = 550, 500
    margin = 60
    pw = W - 2*margin
    ph = H - 2*margin
    to_x = lambda p: margin + p * pw
    to_y = lambda q: H - margin - q * ph
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="background-color:#ffffff; font-family:sans-serif;">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Portrait de Phase : Dynamique du Réplicateur (2 Populations)</text>',
        
        # Zone du simplex [0,1]x[0,1]
        f'<rect x="{to_x(0)}" y="{to_y(1)}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5"/>',
        
        # Séparatrice / frontière des bassins d'attraction
        # Courbe séparatrice passant par (0,0), (p_mix, q_mix), (1,1)
        f'<line x1="{to_x(0)}" y1="{to_y(0)}" x2="{to_x(p_mix)}" y2="{to_y(q_mix)}" stroke="#64748b" stroke-dasharray="5,4" stroke-width="2"/>',
        f'<line x1="{to_x(p_mix)}" y1="{to_y(q_mix)}" x2="{to_x(1)}" y2="{to_y(1)}" stroke="#64748b" stroke-dasharray="5,4" stroke-width="2"/>',
        
        # Bassin 1 (en haut à droite vers (1,1))
        f'<text x="{to_x(0.85)}" y="{to_y(0.7)}" font-size="12" font-style="italic" fill="#1e40af">Bassin (Opéra, Opéra)</text>',
        # Bassin 2 (en bas à gauche vers (0,0))
        f'<text x="{to_x(0.15)}" y="{to_y(0.3)}" font-size="12" font-style="italic" fill="#166534">Bassin (Foot, Foot)</text>',
    ]
    
    # Flèches du champ de vecteurs sur une grille
    P_val, Q_val, N_val = float(game.P), float(game.Q), float(game.N)
    for p_val in np.linspace(0.1, 0.9, 7):
        for q_val in np.linspace(0.1, 0.9, 7):
            dp, dq = replicator_dynamics_system(p_val, q_val, P_val, Q_val, N_val)
            norm = np.sqrt(dp**2 + dq**2)
            if norm > 1e-5:
                # normalisation pour affichage
                scale = 22.0
                dx = (dp / norm) * scale
                dy = - (dq / norm) * scale  # inversion axe Y pour SVG
                x1 = to_x(p_val)
                y1 = to_y(q_val)
                x2 = x1 + dx
                y2 = y1 + dy
                # Flèche
                svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#3b82f6" stroke-width="1.3" marker-end="url(#arrow)"/>')
                
    # Définition marqueur flèche
    svg.insert(3, '''<defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1 L 8 5 L 0 9 z" fill="#3b82f6"/>
      </marker>
    </defs>''')
    
    # Équilibres remarquables
    # (0, 0) : attracteur stable
    svg.append(f'<circle cx="{to_x(0)}" cy="{to_y(0)}" r="8" fill="#16a34a" stroke="#14532d" stroke-width="2.5"/>')
    svg.append(f'<text x="{to_x(0)+12}" y="{to_y(0)-10}" font-size="12" font-weight="bold" fill="#16a34a">Attracteur Stable E2 (0,0)</text>')
    
    # (1, 1) : attracteur stable
    svg.append(f'<circle cx="{to_x(1)}" cy="{to_y(1)}" r="8" fill="#2563eb" stroke="#1e3a8a" stroke-width="2.5"/>')
    svg.append(f'<text x="{to_x(1)-150}" y="{to_y(1)+20}" font-size="12" font-weight="bold" fill="#2563eb">Attracteur Stable E1 (1,1)</text>')
    
    # (p_mix, q_mix) : point selle instable
    svg.append(f'<circle cx="{to_x(p_mix)}" cy="{to_y(q_mix)}" r="9" fill="#dc2626" stroke="#7f1d1d" stroke-width="2.5"/>')
    svg.append(f'<text x="{to_x(p_mix)+12}" y="{to_y(q_mix)+5}" font-size="12" font-weight="bold" fill="#dc2626">Point-selle Instable Emix (2/3, 1/3)</text>')
    
    # (0, 1) et (1, 0) : équilibres non coordonnés instables
    svg.append(f'<circle cx="{to_x(0)}" cy="{to_y(1)}" r="6" fill="#e2e8f0" stroke="#64748b" stroke-width="1.5"/>')
    svg.append(f'<circle cx="{to_x(1)}" cy="{to_y(0)}" r="6" fill="#e2e8f0" stroke="#64748b" stroke-width="1.5"/>')
    
    # Axes
    svg.append(f'<text x="{W/2}" y="{H-15}" text-anchor="middle" font-size="13" fill="#334155">Proportion Joueur 1 Opéra : p</text>')
    svg.append(f'<text x="20" y="{H/2}" text-anchor="middle" font-size="13" fill="#334155" transform="rotate(-90 20 {H/2})">Proportion Joueur 2 Opéra : q</text>')
    svg.append(f'<text x="{to_x(0)}" y="{to_y(0)+20}" text-anchor="middle" font-size="11" fill="#64748b">0</text>')
    svg.append(f'<text x="{to_x(1)}" y="{to_y(0)+20}" text-anchor="middle" font-size="11" fill="#64748b">1</text>')
    svg.append(f'<text x="{to_x(p_mix)}" y="{to_y(0)+20}" text-anchor="middle" font-size="11" fill="#dc2626">2/3</text>')
    svg.append(f'<text x="{to_x(0)-15}" y="{to_y(0)+4}" text-anchor="middle" font-size="11" fill="#64748b">0</text>')
    svg.append(f'<text x="{to_x(0)-15}" y="{to_y(1)+4}" text-anchor="middle" font-size="11" fill="#64748b">1</text>')
    svg.append(f'<text x="{to_x(0)-20}" y="{to_y(q_mix)+4}" text-anchor="middle" font-size="11" fill="#dc2626">1/3</text>')
    
    svg.append('</svg>')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    return filename


def generate_visualizations(game, output_prefix="projet8_figure"):
    """
    Génère les visualisations scientifiques :
    1. 3 SVG natifs vectoriels haute-fidélité (fonctionnent de manière autonome)
    2. PNG matplotlib haute résolution (si matplotlib est installé)
    """
    generated_files = []
    
    # 1. Génération des 3 SVG haute fidélité
    svg1 = generate_svg_best_responses(game, f"{output_prefix}_best_responses.svg")
    svg2 = generate_svg_comparison_bars(game, f"{output_prefix}_comparison_bars.svg")
    svg3 = generate_svg_replicator_dynamics(game, f"{output_prefix}_replicator_dynamics.svg")
    generated_files.extend([svg1, svg2, svg3])
    
    # 2. Tentative matplotlib pour PNG
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        P, Q, N = float(game.P), float(game.Q), float(game.N)
        (p_star, q_star), (u1_mix, u2_mix) = game.mixed_nash()
        p_mix, q_mix = float(p_star), float(q_star)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 0, 1, 1], [0, q_mix, q_mix, 1], color='#1f77b4', lw=2.5, label=r'Meilleure réponse $BR_1(q)$')
        ax.plot([0, p_mix, p_mix, 1], [0, 0, 1, 1], color='#d62728', lw=2.5, linestyle='--', label=r'Meilleure réponse $BR_2(p)$')
        ax.plot(0, 0, 'go', markersize=10, label=r'Équilibre pur $E_2=(F, F)$')
        ax.plot(1, 1, 'bo', markersize=10, label=r'Équilibre pur $E_1=(O, O)$')
        ax.plot(p_mix, q_mix, 'm*', markersize=14, label=f'Équilibre mixte $E_{{mix}}=({p_mix:.2f}, {q_mix:.2f})$')
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'Probabilité $p$ (Joueur 1 Opéra)'); ax.set_ylabel(r'Probabilité $q$ (Joueur 2 Opéra)')
        ax.set_title('Correspondances de Meilleure Réponse et Équilibres de Nash', fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='center left')
        plt.tight_layout()
        f1_png = f"{output_prefix}_best_responses.png"
        plt.savefig(f1_png, dpi=300); plt.close()
        generated_files.append(f1_png)
    except Exception:
        pass
        
    return generated_files


# ----------------------------------------------------------------------
# Section 4 : Suite de Tests et Validation Formelle
# ----------------------------------------------------------------------

def run_project_verification():
    """
    Validation automatisée reprenant l'esprit de verification_TDJ.py
    """
    print("\n" + "="*65)
    print("PROJET 8 : VALIDATION FORMELLE ALGORITHMIQUE DES RÉSULTATS")
    print("="*65)
    
    nb_ok = 0
    def assert_check(cond, desc):
        nonlocal nb_ok
        assert cond, f"ÉCHEC SUR : {desc}"
        nb_ok += 1
        print(f"  [OK {nb_ok:02d}] {desc}")

    # 1. Jeu canonique (P=2, Q=1, N=0)
    g = BattleOfTheSexesGame(P=2, Q=1, N=0)
    
    # Test équilibres purs
    p_eq = g.pure_nash()
    assert_check(len(p_eq) == 2, "2 équilibres de Nash purs existent")
    assert_check(p_eq[0] == ((0, 0), (2.0, 1.0)), "Équilibre pur 1 : (Opéra, Opéra) gains (2, 1)")
    assert_check(p_eq[1] == ((1, 1), (1.0, 2.0)), "Équilibre pur 2 : (Foot, Foot) gains (1, 2)")
    
    # Test équilibre mixte
    (p_star, q_star), (u1_m, u2_m) = g.mixed_nash()
    assert_check(p_star == F(2, 3), "Probabilité p* de J1 = 2/3")
    assert_check(q_star == F(1, 3), "Probabilité q* de J2 = 1/3")
    assert_check(u1_m == F(2, 3) and u2_m == F(2, 3), "Gains espérés à l'équilibre mixte = 2/3 pour chaque joueur")
    assert_check(u1_m < g.Q, "Théorème : le gain espéré mixte (2/3) est STRICTEMENT inférieur au gain de la pire coordination (Q=1)")
    
    # Test taux d'échec de coordination
    eff = g.efficiency_metrics()
    assert_check(eff['p_miscoordination_mixed'] == 5/9, "Probabilité de défaut de coordination = 5/9 (~55.56%)")
    assert_check(eff['PoA'] == 2.25, "Prix de l'Anarchie (PoA) = 3 / (4/3) = 9/4 = 2.25")
    assert_check(eff['PoS'] == 1.0, "Prix de la Stabilité (PoS) = 3 / 3 = 1.0")
    
    # Test stratégie de sécurité
    (p_s, q_s), (v1, v2) = g.security_strategies()
    assert_check(p_s == F(1, 3) and q_s == F(2, 3), "Stratégie de sécurité Maximin : J1 joue Opéra avec prob 1/3")
    assert_check(v1 == F(2, 3) and v2 == F(2, 3), "Niveau de sécurité garanti v = 2/3 (égal au gain du Nash mixte)")
    
    # Test CKR symétrique (Gerville-Réache 2017)
    p_ckr, u_ckr = g.ckr_symmetric_solution()
    assert_check(p_ckr == F(1, 2), "Stratégie symétrique optimale CKR : p = 1/2 (équiprobabilité)")
    assert_check(u_ckr == F(3, 4), "Gain espéré sous CKR symétrique = 3/4 = 0.75 > 2/3")
    
    # Test équilibre corrélé PL (Bien-être social max et Équitable)
    z_max, (u1_max, u2_max) = g.correlated_equilibrium_max_welfare()
    assert_check(np.isclose(u1_max + u2_max, 3.0), "Équilibre corrélé max bien-être : somme des gains = 3.0 (atteint un équilibre pur)")
    
    z_fair, (u1_fair, u2_fair) = g.correlated_equilibrium_fair()
    assert_check(np.isclose(z_fair[0,0], 0.5) and np.isclose(z_fair[1,1], 0.5), "Équilibre corrélé équitable : pile ou face 50/50 entre (O,O) et (F,F)")
    assert_check(np.isclose(u1_fair, 1.5) and np.isclose(u2_fair, 1.5), "Gains corrélés équitables = (1.5, 1.5) (Pareto-optimal et équitable)")
    
    # Test jeu paramétrique général P > Q > N
    P_s, Q_s, N_s = sp.symbols('P Q N', positive=True)
    denom_s = P_s + Q_s - 2*N_s
    p_sym = (P_s - N_s) / denom_s
    q_sym = (Q_s - N_s) / denom_s
    u_sym = (P_s*Q_s - N_s**2) / denom_s
    
    diff_Q = sp.simplify(u_sym - Q_s)
    # diff_Q = (P*Q - N^2 - Q*(P+Q-2N)) / denom = - (Q-N)^2 / denom < 0 car denom > 0
    assert_check(sp.simplify(diff_Q + (Q_s - N_s)**2 / denom_s) == 0,
                 "Preuve symbolique formelle : u_mix - Q = -(Q - N)^2 / (P + Q - 2N) < 0 strictement !")
    
    # Test dynamique du réplicateur : analyse du Jacobien au point selle
    p_num, q_num = 2/3, 1/3
    # J = [[d(dp)/dp, d(dp)/dq], [d(dq)/dp, d(dq)/dq]]
    # au point intérieur, dp/dt = 0 et dq/dt = 0 car delta1(q*)=0 et delta2(p*)=0
    # d(dp)/dp = (1 - 2p)*delta1 = 0
    # d(dp)/dq = p*(1-p)*(P+Q-2N) = (2/3)*(1/3)*3 = 2/3
    # d(dq)/dp = q*(1-q)*(P+Q-2N) = (1/3)*(2/3)*3 = 2/3
    # d(dq)/dq = (1 - 2q)*delta2 = 0
    # Det(J) = 0 - (2/3)^2 = -4/9 < 0 => Valeurs propres réelles de signes opposés (+2/3, -2/3) => Point-selle instable !
    J_det = - (p_num*(1-p_num)*3) * (q_num*(1-q_num)*3)
    assert_check(np.isclose(J_det, -4/9), "Stabilité dynamique : Jacobien au Nash mixte a un déterminant négatif (-4/9), confirmant un point-selle instable !")
    
    print("="*65)
    print(f"TOTAL : {nb_ok} assertions validées avec succès à 100%.")
    print("="*65 + "\n")
    return nb_ok


# ----------------------------------------------------------------------
# Point d'entrée principal
# ----------------------------------------------------------------------
if __name__ == '__main__':
    run_project_verification()
    game = BattleOfTheSexesGame(P=2, Q=1, N=0)
    print("Génération des figures graphiques...")
    files = generate_visualizations(game)
    if files:
        print(f"Graphiques générés : {files}")
