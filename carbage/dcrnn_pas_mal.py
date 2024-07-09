import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Anciennes tranches d'imposition
tranches = [
    (0, 10225, 0.0),  # jusqu'à 10,225€ : 0%
    (10226, 26070, 0.11),  # de 10,226€ à 26,070€ : 11%
    (26071, 74545, 0.30),  # de 26,071€ à 74,545€ : 30%
    (74546, 160336, 0.41),  # de 74,546€ à 160,336€ : 41%
    (160337, float('inf'), 0.45)  # plus de 160,336€ : 45%
]

# Nouvelles tranches d'imposition
nouvelles_tranches = [
    (0, 10292, 0.01),
    (10293, 15438, 0.05),
    (15439, 20584, 0.10),
    (20585, 27789, 0.15),
    (27790, 30876, 0.20),
    (30877, 33964, 0.25),
    (33965, 38081, 0.30),
    (38082, 44256, 0.35),
    (44257, 61752, 0.40),
    (61753, 102921, 0.45),
    (102922, 144089, 0.50),
    (144090, 267594, 0.55),
    (267595, 411683, 0.60),
    (411684, float('inf'), 0.90)
]

# Fonction pour calculer l'impôt en fonction des tranches
def calcul_impot(revenu, tranches):
    impot = 0
    for debut, fin, taux in tranches:
        if revenu > debut:
            impot += (min(revenu, fin) - debut) * taux
        else:
            break
    return impot

# Calcul du taux d'imposition final
revenus = np.arange(1, 400000)
impots = np.array([calcul_impot(revenu, tranches) for revenu in revenus])
taux_imposition = impots / revenus

# Calcul du taux d'imposition final pour les nouvelles tranches
nouveaux_impots = np.array([calcul_impot(revenu, nouvelles_tranches) for revenu in revenus])
nouveau_taux_imposition = nouveaux_impots / revenus

# Calcul du revenu final après impôt
revenu_final_ancien = revenus - impots
revenu_final_nouveau = revenus - nouveaux_impots

# Données INSEE pour ajuster la distribution des revenus
deciles_revenus = [8820, 14010, 16980, 19590, 21980, 24470, 27310, 31020, 37060, 64840]

# Calcul de la moyenne et de l'écart-type approximatifs à partir des déciles
mean_revenu_insee = np.mean(deciles_revenus)
std_revenu_insee = np.std(deciles_revenus)

# Génération des valeurs pour la courbe de distribution des revenus basée sur les données INSEE
frequence_revenus_insee = norm.pdf(revenus, mean_revenu_insee, std_revenu_insee)

# Normalisation pour que le maximum soit égal à 100
frequence_revenus_insee = frequence_revenus_insee / max(frequence_revenus_insee) * 100

# Déterminer les limites des tranches de 10% de la population
percentiles = [norm.ppf(p, mean_revenu_insee, std_revenu_insee) for p in np.arange(0.1, 1.0, 0.1)]
percentiles.insert(0, norm.ppf(0.0, mean_revenu_insee, std_revenu_insee))
percentiles.append(norm.ppf(1.0, mean_revenu_insee, std_revenu_insee))

# Plot
fig, ax1 = plt.subplots(figsize=(14, 8))

ax1.plot(revenus, taux_imposition * 100, label="Anciennes tranches")  # pourcentage
ax1.plot(revenus, nouveau_taux_imposition * 100, label="Nouvelles tranches")  # pourcentage
ax1.plot(revenus, frequence_revenus_insee, label="Fréquence des revenus (INSEE)")  # fréquence des revenus

# Ajouter des couleurs semi-transparentes pour indiquer les tranches de 10% de la population
for i in range(len(percentiles) - 1):
    ax1.fill_between(revenus, 0, frequence_revenus_insee, where=(revenus >= percentiles[i]) & (revenus <= percentiles[i + 1]), alpha=0.3, label=f'{int(i*10)}%-{int((i+1)*10)}% de la population')

ax1.set_title("Comparaison des taux d'imposition, de la fréquence des revenus et du revenu final en France (jusqu'à 100 000€)")
ax1.set_xlabel("Revenu (€)")
ax1.set_ylabel("Taux d'imposition (%) / Fréquence des revenus (%)")
ax1.legend(loc='upper left')
ax1.grid(True)
ax1.set_xlim(0, 100000)
ax1.set_ylim(0, 100)

# Annotation des nouvelles tranches
for debut, fin, taux in nouvelles_tranches:
    if debut <= 100000:
        ax1.axvline(x=debut, color='grey', linestyle='--', linewidth=0.5)
        if fin != float('inf') and fin <= 100000:
            ax1.axvline(x=fin, color='grey', linestyle='--', linewidth=0.5)

# Ajouter un second axe y pour le revenu final
ax2 = ax1.twinx()
ax2.plot(revenus, revenu_final_ancien, color='red', label="Revenu final après impôt (Anciennes tranches)")
ax2.plot(revenus, revenu_final_nouveau, color='blue', label="Revenu final après impôt (Nouvelles tranches)")
ax2.set_ylabel("Revenu final (€)")
ax2.legend(loc='upper right')

plt.show()
