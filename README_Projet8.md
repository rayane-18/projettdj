# Projet 8 — Jeux de Coordination (La Bataille des Sexes)
**Master 2 Mathématiques $\cdot$ Recherche Opérationnelle (RO) — Université Akli Mohand Oulhadj - Bouira**  
**Module : Théorie des Jeux (Année Universitaire 2025/2026)**

---

## 🎯 Objectifs et Cadre du Projet
- **Sujet :** Projet 8 — Jeux de coordination (La Bataille des Sexes / *Bach or Stravinsky*).
- **Objectif :** Analyser les jeux à équilibres multiples et le problème de la sélection d'équilibre.
- **Livrables :** Identification formelle des équilibres purs et mixtes, discussion approfondie des points focaux (Thomas Schelling), analyse épistémique (CKR / HAL 2017), équilibre corrélé (Robert Aumann) et dynamique évolutive du réplicateur (Maynard Smith).
- **Piste de départ :** Comparer les équilibres et leurs gains espérés respectifs, et démontrer le théorème de défaillance de coordination (*coordination failure*).

---

## 📂 Contenu du Répertoire du Projet

| Fichier | Description |
| :--- | :--- |
| **`Projet8_Jeux_Coordination_Bataille_Sexes.pdf`** | **Mémoire universitaire complet (26 pages)** compilé avec pdfLaTeX (MiKTeX), comportant page de garde officielle, introduction dédiée, table des matières et 8 sections détaillées. |
| **`Projet8_Jeux_Coordination_Bataille_Sexes.tex`** | Code source LaTeX modulaire avec environnements `tcolorbox`, diagrammes `TikZ` (arbre séquentiel, meilleures réponses), démonstrations formelles et bibliographie académique. |
| **`projet8_coordination.py`** | **Module d'analyse formelle et numérique** : calcul symbolique (`sympy`), optimisation linéaire HiGHS (`scipy.optimize.linprog`), calcul de jacobienne, simulations Monte-Carlo (50 000 itérations) et validation à 100% (**19 assertions vérifiées**). |
| **`dashboard_coordination.html`** | **Application Web interactive (HTML5 / SVG / JS)** : paramétrage dynamique des matrices $(P, Q, N)$, tracé temps réel des correspondances de meilleure réponse et simulation empirique. |
| `projet8_figure_best_responses.svg` | Graphique vectoriel des correspondances $BR_1(q)$ et $BR_2(p)$ et des trois équilibres de Nash. |
| `projet8_figure_replicator_dynamics.svg` | Portrait de phase vectoriel de la dynamique du réplicateur (point-selle instable au Nash mixte et bassins d'attraction). |
| `projet8_figure_comparison_bars.svg` | Histogramme comparatif de l'ensemble des concepts de solution (Purs, Mixte, Sécurité, CKR, Corrélé, Répété). |

---

## 🧭 Structure et Organisation du Mémoire (26 pages)

1. **Page de Garde Officielle** : Université Akli Mohand Oulhadj -- Bouira, Département de Mathématiques / RO.
2. **Introduction Générale (sur sa propre page, pagination romaine)** :
   - Présentation narrative de la Bataille des Sexes (l'arbitrage entre passer la soirée ensemble et les préférences divergentes pour le Football vs l'Opéra).
   - **La Problématique Centrale :** comment des agents rationnels sans communication sélectionnent-ils un équilibre parmi les équilibres multiples sans provoquer le désaccord ?
   - Articulation détaillée et annonce du plan des 8 sections du mémoire.
3. **Table des Matières Détaillée** (`\tableofcontents`).
4. **Section 1 : Modélisation Formelle et Typologie des Jeux de Coordination** :
   - Cadre axiomatique sous forme normale $\Gamma = \langle I, (S_i), (u_i) \rangle$.
   - Taxonomie des jeux $2 \times 2$ (Dilemme du Prisonnier, Somme Nulle, Pure Coordination, Chasse au Cerf, Bataille des Sexes).
   - Paramétrisation générale $P > Q > N \ge 0$ (notes de cours MITRO206) et forme canonique $(2, 1)/(1, 2)/(0, 0)$.
5. **Section 2 : Identification Exhaustive des Équilibres de Nash** :
   - Dérivation formelle des deux équilibres purs $E_1=(O, O)$ et $E_2=(F, F)$ (Pareto-optimaux mais dissymétriques).
   - Dérivation pas à pas de l'équilibre mixte par indifférence au support : $p^* = \frac{P-N}{P+Q-2N} = 2/3$, $q^* = \frac{Q-N}{P+Q-2N} = 1/3$.
   - Paradoxe d'inversion des incitations de Nash.
   - Diagramme TikZ de l'intersection des correspondances de meilleure réponse dans $[0, 1]^2$.
6. **Section 3 : Comparaison Analytique des Équilibres et Défaillance de Coordination** :
   - Démonstration du théorème de sous-optimalité stricte : $u_{mix}^* - Q = -\frac{(Q-N)^2}{P+Q-2N} < 0 \implies \mathbf{u_{mix}^* < Q < P}$.
   - Quantification du risque de désaccord : $\mathbb{P}(\text{défaut}) = 5/9 \approx 55{,}56\%$.
   - Calcul du Prix de l'Anarchie ($\text{PoA} = 2{,}25$) et du Prix de la Stabilité ($\text{PoS} = 1{,}00$).
7. **Section 4 : Critiques Épistémiques et Concepts Alternatifs de Solution** :
   - Critique de la Connaissance Commune de la Rationalité (CKR, étude de L. Gerville-Réache, HAL 2017).
   - Stratégie de sécurité Maximin de von Neumann : $p_{sec} = 1/3$ garantissant $v = 2/3$ (strictement égal au gain du Nash mixte).
   - Issue symétrique collective sous CKR : choix équiprobable ($p=1/2$) procurant un gain de $3/4 = 0{,}75 > 0{,}667$.
   - Équilibre Corrélé de Robert Aumann (1974) : signal public $50/50$, gains $(1{,}50, 1{,}50)$, Pareto-optimal et équitable.
8. **Section 5 : Sélection d'Équilibre et Points Focaux de Schelling** :
   - Théorie des points focaux (Thomas Schelling, 1960, *The Strategy of Conflict*) et saillance contextuelle.
   - Communication préalable non engageante (*cheap talk*).
   - Engagement irréversible et leadership de Stackelberg (arbre séquentiel TikZ et induction à rebours).
   - Conventions temporelles d'alternance répétée (*turn-taking* et théorème Folk).
   - Analyse dynamique du réplicateur : matrice jacobienne avec $\det(J) = -4/9 < 0$ prouvant que l'équilibre mixte est un **point-selle instable**.
9. **Section 6 : Applications Réelles et Recherche Opérationnelle** :
   - Guerres de standards technologiques (VHS vs Betamax, Blu-Ray vs HD-DVD, connecteur USB-C).
   - Blocages budgétaires et *shutdowns* gouvernementaux américains (Chapitre 15 du cours).
   - Choix de plateformes technologiques conjointes en entreprise (MAC vs CAM).
10. **Section 7 : Synthèse Comparative et Validation Algorithmique** :
    - Tableau synoptique de toutes les solutions.
    - Description de l'architecture logicielle et des 19 tests unitaires validés à 100%.
    - Présentation des fonctionnalités du tableau de bord interactif.
11. **Section 8 : Conclusion Générale et Perspectives** :
    - Bilan conceptuel et prolongements (Jeux globaux avec information incomplète, réseaux de coordination).
12. **Références Bibliographiques Exhaustives** (15 références académiques majeures).

---

## 📊 Tableau Synoptique Comparatif des Solutions

| Concept de Solution | Stratégies $(s_1, s_2)$ | Gain $u_1$ | Gain $u_2$ | Bien-être Social | Taux de Succès | Propriété Clé |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Nash Pur 1 ($E_1$)** | $(O, O)$ | $2{,}00$ | $1{,}00$ | $3{,}00$ | $100\%$ | Pareto-optimal, asymétrique J1 |
| **Nash Pur 2 ($E_2$)** | $(F, F)$ | $1{,}00$ | $2{,}00$ | $3{,}00$ | $100\%$ | Pareto-optimal, asymétrique J2 |
| **Nash Mixte** | $p^*=2/3, q^*=1/3$ | $\mathbf{0{,}67}$ | $\mathbf{0{,}67}$ | $\mathbf{1{,}33}$ | $\mathbf{44{,}4\%}$ | **Destructeur de valeur ($u < Q$)** |
| **Sécurité (Maximin)** | $p=1/3, q=2/3$ | $0{,}67$ | $0{,}67$ | $1{,}33$ | $44{,}4\%$ | Gain garanti sans confiance ($v = 2/3$) |
| **Symétrique CKR (HAL 2017)** | $p=1/2$ (équiprobable) | $0{,}75$ | $0{,}75$ | $1{,}50$ | $50{,}0\%$ | Maximise $\max 3p(1-p)$ sous CKR |
| **Équilibre Corrélé (Aumann)**| Lancer de pièce $50/50$ | $\mathbf{1{,}50}$ | $\mathbf{1{,}50}$ | $\mathbf{3{,}00}$ | $\mathbf{100\%}$ | **Pareto-dominant & équitable** |
| **Alternance Répétée (Folk)** | Jour pair $O$ / impair $F$ | $\mathbf{1{,}50}$ | $\mathbf{1{,}50}$ | $\mathbf{3{,}00}$ | $\mathbf{100\%}$ | Convention sociale endogène |

---

## 🚀 Guide d'Exécution et Consultation

```powershell
# 1. Consulter le mémoire PDF universitaire complet (26 pages)
Start-Process "Projet8_Jeux_Coordination_Bataille_Sexes.pdf"

# 2. Exécuter la suite de validation formelle en Python (19 assertions)
python projet8_coordination.py

# 3. Lancer le tableau de bord interactif dans votre navigateur Web
Start-Process "dashboard_coordination.html"
```
