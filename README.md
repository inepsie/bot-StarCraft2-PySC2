# bot-StarCraft2-PySC2

Programme qui joue à StarCraft2 via le wrapper PySC2 (effectué en binôme avec Emmanuel Mompi @emompi).

Le joueur-ordi de race Zerg effectue une stratégie de "Rush Roaches". Cette dernière consiste à envoyer le plus rapidement possible des vagues d'unités de type "Roaches". Cependant notre programme permet en fait de construire n'importe quelle unité et bâtiment de la race Zerg.

Nous avons utilisé un FSM (Finite State Machine) via la librairie Fysom. Certaines étapes possèdent plusieurs dépendances, exemple : pour construire une unité A, il faut un bâtiment B + avoir débloqué la technologie C ect. En plus du FSM, un "techtree" contenant les dépendances et les matérieux nécessaire nous aide à organiser notre production.

Ci-dessous une description un peu plus précise :
![FSM](./Rapport/Capture.JPG?raw=true)

Nous avons évalué les performances de notre bot contre les IA du jeu. Les pourcentages de victoire sont relativement proportionnels à la difficulté de l'adversaire :

![performances de notre bot](./Rapport/resultats.JPG?raw=true)


