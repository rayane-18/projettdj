# Projet 8 — Jeux de Coordination (La Bataille des Sexes)
**Master 1 Recherche Opérationnelle (RO) — Université Akli Mohand Oulhadj - Bouira**  
**Module : Théorie des Jeux (2025/2026)**

---

## 🎯 Objectifs et Cadre du Projet
- **Sujet :** Projet 8 — Jeux de coordination (bataille des sexes).
- **Objectif :** Analyser les jeux à équilibres multiples et le problème de la sélection d'équilibre.
- **Livrables :** Identification formelle des équilibres purs et mixtes, discussion approfondie des points focaux (Thomas Schelling).
- **Piste de départ :** Comparer les équilibres et leurs gains espérés respectifs.

---

## 📂 Contenu du Répertoire du Projet

| Fichier | Description |
| :--- | :--- |
| **`Projet8_Jeux_Coordination_Bataille_Sexes.pdf`** | **Rapport académique complet (12 pages)** compilé avec pdfLaTeX, conforme aux chartes du Master RO de Bouira. |
| **`Projet8_Jeux_Coordination_Bataille_Sexes.tex`** | Code source LaTeX avec boîtes `tcolorbox`, figures `TikZ`, démonstrations formelles et bibliographie. |
| **`projet8_coordination.py`** | **Script de modélisation, calcul symbolique (`sympy`), optimisation linéaire (`scipy`), simulations Monte-Carlo et validation algorithmique** (19/19 tests réussis). |
| **`dashboard_coordination.html`** | **Tableau de bord interactif (HTML5 / SVG / JS)** permettant de modifier dynamiquement les gains $(P, Q, N)$, recalculer les équilibres et simuler 50 000 parties en temps réel. |
| `projet8_figure_best_responses.svg` | Graphique vectoriel des correspondances de meilleure réponse $BR_1(q)$ et $BR_2(p)$ et des 3 équilibres de Nash. |
| `projet8_figure_replicator_dynamics.svg` | Portrait de phase vectoriel de la dynamique du réplicateur (point-selle instable au Nash mixte et bassins d'attraction). |
| `projet8_figure_comparison_bars.svg` | Histogramme comparatif de tous les concepts de solutions (purs, mixte, sécurité, CKR, corrélé, alternance). |

---

## 🔬 Synthèse des Résultats Mathématiques et Conceptuels

### 1. Modélisation Paramétrique Générale ($P > Q > N \ge 0$)
Considérons la matrice bimatricielle générale (notes MITRO206, section 1.3.6) :
$$
\begin{pmatrix}
(P, Q) & (N, N) \\
(N, N) & (Q, P)
\end{pmatrix}
\quad \xrightarrow{\text{Cas canonique}} \quad
\begin{pmatrix}
(2, 1) & (0, 0) \\
(0, 0) & (1, 2)
\end{pmatrix}
$$

### 2. Équilibres de Nash
1. **Deux équilibres en stratégies pures :**
   - $E_1 = (O, O)$ avec gains $(P, Q) = (2, 1)$ — favorise le Joueur 1.
   - $E_2 = (F, F)$ avec gains $(Q, P) = (1, 2)$ — favorise le Joueur 2.
   - Les deux équilibres sont Pareto-optimaux mais asymétriques.
2. **Un unique équilibre en stratégies mixtes intérieur :**
   $$p^* = \frac{P - N}{P + Q - 2N} = \frac{2}{3}, \qquad q^* = \frac{Q - N}{P + Q - 2N} = \frac{1}{3}$$
   - *Propriété paradoxale :* chaque joueur randomise selon les gains de son partenaire pour assurer son indifférence.

### 3. Théorème de la Défaillance de Coordination (*Coordination Failure*)
L'espérance de gain à l'équilibre mixte est strictement identique pour les deux agents :
$$u_1^* = u_2^* = \frac{PQ - N^2}{P + Q - 2N} = \frac{2}{3} \approx 0{,}667$$

**Preuve formelle de sous-optimalité :**
$$u_{mix}^* - Q = - \frac{(Q - N)^2}{P + Q - 2N} < 0 \implies \mathbf{u_{mix}^* < Q < P}$$
Le gain de l'équilibre mixte ($2/3$) est **strictement inférieur au gain le plus faible en cas de coordination** ($Q=1$).
- **Probabilité de défaut de coordination :**
  $$\mathbb{P}(\text{désaccord}) = p^*(1 - q^*) + (1 - p^*)q^* = \frac{5}{9} \approx 55{,}56\%$$
  Plus d'une fois sur deux, des agents rationnels sans communication échouent à se synchroniser !
- **Price of Anarchy (PoA) :**
  $$\text{PoA} = \frac{W_{opt}}{W(E_{mix})} = \frac{3}{4/3} = 2{,}25$$

---

## 📊 Tableau Comparatif des Concepts de Solution

| Concept de Solution | Profil de Stratégies | Gain J1 | Gain J2 | Bien-Être Social | Taux de Succès | Statut / Propriété |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Nash Pur 1 ($E_1$)** | $(O, O)$ | $2{,}00$ | $1{,}00$ | $3{,}00$ | $100\%$ | Pareto-optimal, asymétrique J1 |
| **Nash Pur 2 ($E_2$)** | $(F, F)$ | $1{,}00$ | $2{,}00$ | $3{,}00$ | $100\%$ | Pareto-optimal, asymétrique J2 |
| **Nash Mixte** | $p^*=2/3, q^*=1/3$ | $0{,}67$ | $0{,}67$ | $1{,}33$ | $44{,}4\%$ | **Destructeur de valeur ($u < Q$)** |
| **Sécurité (Maximin)** | $p=1/3, q=2/3$ | $0{,}67$ | $0{,}67$ | $1{,}33$ | $44{,}4\%$ | Gain garanti sans confiance ($v = 2/3$) |
| **Symétrique CKR (HAL 2017)** | $p=1/2$ (équiprobable) | $0{,}75$ | $0{,}75$ | $1{,}50$ | $50{,}0\%$ | Maximise $\max 3p(1-p)$ sous CKR |
| **Équilibre Corrélé (Aumann)**| Lancer de pièce $50/50$ | $\mathbf{1{,}50}$ | $\mathbf{1{,}50}$ | $\mathbf{3{,}00}$ | $\mathbf{100\%}$ | **Pareto-dominant & équitable** |
| **Alternance Répétée (Folk)** | Jour pair $O$ / impair $F$ | $\mathbf{1{,}50}$ | $\mathbf{1{,}50}$ | $\mathbf{3{,}00}$ | $\mathbf{100\%}$ | Convention sociale endogène |

---

## 🧭 Le Problème de la Sélection d'Équilibre et les Points Focaux de Schelling

1. **Points Focaux (Thomas Schelling, 1960) :**
   - Saillance cognitive, culturelle ou contextuelle permettant une convergence tacite des anticipations sans communication.
2. **Communication préalable (*Cheap Talk*) :**
   - Crawford \& Sobel (1982), Farrell (1987) : une déclaration unilatérale préalable est crédible car les intérêts sont alignés sur la coordination.
3. **Engagement séquentiel et leadership de Stackelberg (Manuel de cours, Chapitre 15) :**
   - Acheter un billet non remboursable transforme le jeu simultané en jeu séquentiel où le suiveur rationnel capitule pour maximiser son gain résiduel.
4. **Instabilité dynamique (Dynamique du réplicateur) :**
   - La matrice jacobienne au Nash mixte possède $\det(J) = -4/9 < 0$, prouvant que l'équilibre mixte est un **point-selle instable**. Toute perturbation conduit la population vers l'une des deux conventions pures.

---

## ⚙️ Exécution et Vérification

### 1. Exécuter la suite de validation Python :
```bash
python projet8_coordination.py
```
*Sortie attendue :* `TOTAL : 19 assertions validées avec succès à 100%.`

### 2. Ouvrir le tableau de bord interactif :
Double-cliquez sur `dashboard_coordination.html` ou ouvrez-le dans n'importe quel navigateur Web moderne pour tester des matrices arbitraires et lancer des simulations Monte-Carlo.

### 3. Consulter le rapport PDF :
Ouvrez `Projet8_Jeux_Coordination_Bataille_Sexes.pdf` pour la lecture complète du mémoire universitaire.
