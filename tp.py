'''

Le debut du bot par:

Theo Montaigu (18907784)

Emmanuel Mompi (18901914)


'''

import pandas as pd
import os
import sys


import random
import numpy as np
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop
from fysom import Fysom

from updatedtree import techtree
from actions import info_actions
from empty_agent import empty_agent2



# df_wins = pd.DataFrame()
# bot_difficulty = None





class MyAgent(base_agent.BaseAgent):
  # one-time setup
  def __init__(self):
    super(MyAgent, self).__init__()

  # read state from obs and ALWAYS act (it's necessary)
  def step(self, obs):
    super(MyAgent, self).step(obs)

    return actions.FUNCTIONS.no_op()


class ZergAgent(base_agent.BaseAgent):

    def __init__(self):
        super(ZergAgent, self).__init__()
        self.step_acc = 1
        self.id_attack = 0
        self.base_top_left = None
        self.build_queue = [89]
        self.build_completed = []
        self.build_work_in_progress = []
        self.SCREEN_DIM_XY = 29
        self.RAW_RES = 182
        self.FSM = None
        self.BO = None
        self.center_checked_to_find_XY = []
        self.current_build_train = None
        self.camera_coord = None
        self.building_coord = []
        self.expand_coord = [(118, 60), (47, 137)]
        self.gas_harvesters = []
        self.first_bases = [(112, 129), (54, 67)]
        self.third_hatch = [(93, 141), (78, 73)]

        self.step_count = 0

    def units_in_progress(self, obs, id):
        units = self.get_my_units_by_type(obs, id)
        for unit in units:
            if unit.build_progress < 100:
                return True
        units = self.get_my_units_by_type(
            obs, techtree[id]["morph_from_id"][0])
        for unit in units:
            if unit.order_id_0 == info_actions[id]["order"]:
                return True
        return False

    def building_exists(self, obs, id):
        buildings = self.get_my_units_by_type(obs, id)
        if(len(buildings) == 0):
            return False
        return True

    def can_build(self, obs, id):
        # can_build() return True si on possède tout ce qui est nécessaire à la création de l'unit "id".
        # Si certaines units manquent, can_build return le requierement le plus fondamental.
        # Exemple: Roach needs Warren_Roach,  Warren_roach needs Spawning_pool, Spawning_pool needs Hatchery.
        # Si on appel can_build(Roach), qu'on ne possède pas de Warren_roach et pas de Spawning_pool mais qu'on possède une Hatchery :
        # can_build(Roach) va return l'id de la Spawning_pool pour qu'on lance sa construction.

        # techtree[id]['requirement'][0] et techtree[id]['morph_from_id'][0] sont directement nécessaire pour la création de "id".
        # Les autres éléments de techtree[id]['requirement'] et de techtree[id]['morph_from_id'][0] sont indirectement nécessaire.
        # Par exemple, pour train un baneling on a directement besoin d'un zergling et d'une Baneling_nest. Ces deux éléments sont nécessaires et suffisants.
        # Cependant le techtree nous indiquera aussi les dépendances indirectes.

        # On possède les requirements nécessaires et suffisants:
        if self.building_exists(obs, techtree[id]['requirement'][0]) and self.building_exists(obs, techtree[id]['morph_from_id'][0]):
            # print (f"Can build : {techtree[id]['name']}")
            return True

        # Il nous manque des dépendances, on return la dépendances la plus fondamentale comme expliqué précedemment:
        if len(techtree[id]['requirement']) > 0:
            check_against_req = [item for item in techtree[id]
                ['requirement'] if not self.building_exists(obs, item)]
            if len(check_against_req) > 0:
                # print (f"Can't build : {techtree[id]['name']}")
                # print(f"Missing : {techtree[check_against_req[0]]['name']}\n")
                return check_against_req[0]

        # Il nous manque des dépendances, on return la dépendances la plus fondamentale comme expliqué précedemment:
        if len(techtree[id]['morph_from_id']) > 0:
            check_against_morph = [item for item in techtree[id]
                ['morph_from_id'] if not self.building_exists(obs, item)]
            if len(check_against_morph) > 0:
                # print (f"Can't build : {techtree[id]['name']}")
                # print(f"Missing : {techtree[check_against_morph[0]]['name']}\n")
                return check_against_morph[0]

        # print (f"Can build : {techtree[id]['name']}")
        return True

    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.SELF]

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def can_do(self, obs, action):
        return action in obs.observation['available_actions']

    def reset(self):
        super(ZergAgent, self).reset()
        self.step_acc = 1
        self.id_attack = 0
        self.base_top_left = None
        self.build_queue = [89]
        self.build_completed = []
        self.build_work_in_progress = []
        self.SCREEN_DIM_XY = 29
        self.RAW_RES = 182
        self.FSM = None
        self.BO = None
        self.center_checked_to_find_XY = []
        self.current_build_train = None
        self.camera_coord = None
        self.building_coord = []
        self.expand_coord = [(118, 60), (47, 137)]
        self.gas_harvesters = []
        self.first_bases = [(112, 129), (54, 67)]
        self.third_hatch = [(93, 141), (78, 73)]

        self.step_count = 0

        self.FSM = Fysom({"initial": {"state": "ready"},
    "final": "end",
    "events": [
        {"name": "train_wanted", "src": "ready", "dst": "train_unit"},
        {"name": "building_wanted", "src": [
            "ready", "train_unit"], "dst": "build"},
        {"name": "done", "src": "train_unit", "dst": "ready"},
        {"name": "done", "src": "build", "dst": "ready"},
        {"name": "done", "src": "find_center", "dst": "ready"},
        {"name": "done", "src": "find_coord", "dst": "ready"},
        {"name": "center_wanted", "src": "build", "dst": "find_center"},
        {"name": "coord_wanted", "src": "find_center", "dst": "find_coord"},
        {"name": "center_wanted", "src": "find_coord", "dst": "find_center"}
        ]})

        self.BO = Fysom({"initial": {"state": "phase_0"},
    "final": "end",
    "events": [
        {"name": "p0_to_p1", "src": "phase_0", "dst": "phase_1"},
        {"name": "p1_to_p2", "src": "phase_1", "dst": "phase_2"},
        {"name": "p2_to_p3", "src": "phase_2", "dst": "phase_3"},
        {"name": "p3_to_p4", "src": "phase_3", "dst": "phase_4"},
        {"name": "p4_to_p5", "src": "phase_4", "dst": "phase_5"},
        {"name": "p5_to_p6", "src": "phase_5", "dst": "phase_6"},
        {"name": "p6_to_p7", "src": "phase_6", "dst": "phase_7"},
        {"name": "p7_to_p8", "src": "phase_7", "dst": "phase_8"},
        {"name": "p8_to_p7", "src": "phase_8", "dst": "phase_7"}
        ]})



        print(f'Roach number : {ROACH_NUMBER}')
        



    def nb_units_in_progress(self, obs, id):
        in_progress = []
        if(techtree[id]["is_building"]):
            units = self.get_my_units_by_type(obs, id)
            in_progress = [elmt for elmt in units if (
                elmt.build_progress < 100)]
            return len(in_progress)
        cocoons = self.get_my_units_by_type(obs, 103)
       # print("ORDER COCOONS")
        for cocoon in cocoons:
           # print(cocoon.order_id_0)
            in_progress = [elmt for elmt in cocoons if elmt.order_id_0 == info_actions[id]["order"]]
       # print("IN_PROGRESS")
       # print(len(in_progress))
        return len(in_progress)

    # 1/ Au commencement de la partie on modifie certaines informations en fonction d'où se situe notre première Hatch
    # 2/ On train un Drone

    def phase_0(self, obs):
       # print("PHASE 0")
        hatch = self.get_my_units_by_type(obs, units.Zerg.Hatchery)[0]
        if(hatch.x > 40 and hatch.x < 70):
            self.expand_coord.reverse()
            self.first_bases.reverse()
            self.third_hatch.reverse()
        larva = self.get_my_units_by_type(obs, units.Zerg.Larva)[0]
        self.BO.p0_to_p1()
        return actions.RAW_FUNCTIONS.Train_Drone_quick("now", larva)

    # On train des Drones jusqu'à en avoir 14. Ensuite on build un Extractor
    def phase_1(self, obs):
       # print("PHASE 1")
        drones = self.get_my_units_by_type(obs, units.Zerg.Drone)
        if(len(drones) > 13):
            self.BO.p1_to_p2()
            return self.build_extractor(obs)
        larvas = self.get_my_units_by_type(obs, units.Zerg.Larva)
        if(len(larvas) > 0):
            return actions.RAW_FUNCTIONS.Train_Drone_quick("now", larvas[0])
        return actions.RAW_FUNCTIONS.no_op()

    # Quand l'Extractor a fini sa construction on met 3 Drones dessus
    # Tant qu'on a moins de 14 Drones on en train
    # Si on possède 14 Drones, on a atteint la limite max de pop donc on train un Overlord
    def phase_2(self, obs):
       # print("PHASE 2")
        if(self.need_more_drones_on_extractor(obs)):
            return self.put_drones_on_extractor(obs)
        drones = self.get_my_units_by_type(obs, units.Zerg.Drone)
        if(len(drones) < 14):
            larvas = self.get_my_units_by_type(obs, units.Zerg.Larva)
            if(len(larvas) > 0):
                return actions.RAW_FUNCTIONS.Train_Drone_quick("now", larvas[0])
            return actions.RAW_FUNCTIONS.no_op()
        if(obs.observation.player.minerals > 100):
            larvas = self.get_my_units_by_type(obs, units.Zerg.Larva)
            if(len(larvas) > 0):
                self.BO.p2_to_p3()
                self.FSM.train_wanted()
                return self.train(obs, 106, 1)
        return actions.RAW_FUNCTIONS.no_op()

    # Si on a 200 de minerais on build une Spawning Pool
    # Sinon on Train des Drones
    def phase_3(self, obs):
       # print("PHASE 3")
        if(self.need_more_drones_on_extractor(obs)):
            return self.put_drones_on_extractor(obs)
        if(obs.observation.player.minerals > 200):
            self.BO.p3_to_p4()
            self.FSM.building_wanted()
            return self.build(obs, 89, 1)
        larvas = self.get_my_units_by_type(obs, units.Zerg.Larva)
        if(len(larvas) > 0):
            return actions.RAW_FUNCTIONS.Train_Drone_quick("now", larvas[0])
        return actions.RAW_FUNCTIONS.no_op()

    # Si on a 300 de minerais on build une Hatchery
    # Sinon on Train des Drones
    def phase_4(self, obs):
       # print("PHASE 4")
        if(self.need_more_drones_on_extractor(obs)):
            return self.put_drones_on_extractor(obs)
        if(obs.observation.player.minerals > 300):
            self.BO.p4_to_p5()
            self.FSM.building_wanted()
            return self.build(obs, 86, 1)
        larvas = self.get_my_units_by_type(obs, units.Zerg.Larva)
        if(len(larvas) > 0):
            distances = self.get_distances(obs, larvas, self.first_bases[0])
            larva = larvas[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Train_Drone_quick("now", larva)
        return actions.RAW_FUNCTIONS.no_op()

    def phase_5(self, obs):
       # print("PHASE 5")
        if(self.need_more_drones_on_extractor(obs)):
            return self.put_drones_on_extractor(obs)
        if(obs.observation.player.minerals > 50 and len(self.get_my_units_by_type(obs, units.Zerg.Extractor)) < 2):
            return self.build_extractor(obs)
        if(obs.observation.player.minerals > 150):
            self.BO.p5_to_p6()
            self.FSM.building_wanted()
            return self.build(obs, 97, 1)
        larvas = self.get_my_units_by_type(obs, units.Zerg.Larva)
        if(len(larvas) > 0):
            distances = self.get_distances(obs, larvas, self.first_bases[0])
            larva = larvas[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Train_Drone_quick("now", larva)
        return actions.RAW_FUNCTIONS.no_op()

    def phase_6(self, obs):
       # print("PHASE 6")
        if(self.need_more_drones_on_extractor(obs)):
            return self.put_drones_on_extractor(obs)
        drones = self.get_my_units_by_type(obs, units.Zerg.Drone)
        if(obs.observation.player.minerals > 300):
            hatcheries = self.get_my_units_by_type(obs, units.Zerg.Hatchery)
            if(len(drones) > 0):
                self.BO.p6_to_p7()
                return actions.RAW_FUNCTIONS.Build_Hatchery_pt('now', drones[0].tag, self.third_hatch[0])
        if((self.nb_units_in_progress(obs, 104) + len(drones)) < 24):
            larvas = self.get_my_units_by_type(obs, units.Zerg.Larva)
            if(len(larvas) > 0):
                distances = self.get_distances(
                    obs, larvas, self.first_bases[0])
                larva = larvas[np.argmin(distances)]
                return actions.RAW_FUNCTIONS.Train_Drone_quick("now", larva)
        self.FSM.train_wanted()
        return self.train(obs, 110, 1)

    def phase_7(self, obs):
       # print("PHASE 7")
        if(self.need_more_drones_on_extractor(obs)):
           # print("A")
            return self.put_drones_on_extractor(obs)
        Roaches = self.get_my_units_by_type(obs, 110)
        drones = self.get_my_units_by_type(obs, units.Zerg.Drone)
        if((self.nb_units_in_progress(obs, 104) + len(drones)) < 24):
           # print("B")
            larvas = self.get_my_units_by_type(obs, units.Zerg.Larva)
            if(len(larvas) > 0):
                distances = self.get_distances(
                    obs, larvas, self.first_bases[0])
                larva = larvas[np.argmin(distances)]
                return actions.RAW_FUNCTIONS.Train_Drone_quick("now", larva)
                if(len(Roaches) > ROACH_NUMBER):
                    self.BO.p7_to_p8()
                return actions.RAW_FUNCTIONS.Train_Drone_quick("now", larvas[0])
        if((obs.observation.player.food_cap - obs.observation.player.food_used + 8 * self.nb_units_in_progress(obs, 106)) < 14):
           # print("C")
            self.FSM.train_wanted()
            if(len(Roaches) > ROACH_NUMBER):
                self.BO.p7_to_p8()
            return self.train(obs, 106, 1)
        if(len(Roaches) > ROACH_NUMBER):
            self.BO.p7_to_p8()
        self.FSM.train_wanted()
        return self.train(obs, 110, 1)

    def phase_8(self, obs):
        Roaches = self.get_my_units_by_type(obs, 110)
        if(len(Roaches) > ROACH_NUMBER):
            Roaches_tag = [elmt.tag for elmt in Roaches]
            self.BO.p8_to_p7()
            if(self.id_attack == 0):
                return actions.RAW_FUNCTIONS.Attack_pt("now", Roaches_tag, self.first_bases[1])
            return actions.RAW_FUNCTIONS.Attack_pt("now", Roaches_tag, self.expand_coord[0])
        self.BO.p8_to_p7()
        return actions.RAW_FUNCTIONS.no_op()

    '''
    # 1/Première action qu'on lance afin de directement construire un drone.
    # 2/On choisis l'endroit de la première expand en fonction du placement de la première hatch
    def start(self, obs):
        hatch = self.get_my_units_by_type(obs, units.Zerg.Hatchery)[0]
        if(hatch.x == 112):
            self.expand_coord.reverse()
        larva = self.get_my_units_by_type(obs,units.Zerg.Larva)[0]
        self.FSM.drone_wanted()
        return actions.RAW_FUNCTIONS.Train_Drone_quick("now", larva)'''

    # Construction d'un extractor sur le gaz le plus proche de la première hatchery
    def build_extractor(self, obs):
        if(obs.observation.player.minerals > 25):
            hatcherie = self.get_my_units_by_type(obs, units.Zerg.Hatchery)[0]
            drone = self.get_my_units_by_type(obs, units.Zerg.Drone)[0]
            extractors = self.get_my_units_by_type(obs, units.Zerg.Extractor)
            extra_coord = [(extra.x, extra.y) for extra in extractors]
            vespenes = [vespene for vespene in obs.observation.raw_units if vespene.unit_type == 342 and (
                vespene.x, vespene.y) not in extra_coord]
            distances = self.get_distances(obs, vespenes, self.first_bases[0])
            vespene = vespenes[np.argmin(distances)]
            # print(vespene.x, vespene.y)
            self.build_completed.append(units.Zerg.Extractor)
            return actions.RAW_FUNCTIONS.Build_Extractor_unit('now', drone.tag, vespene.tag)
        return actions.RAW_FUNCTIONS.no_op()

    # Permet de savoir si on lance put_drones_on_extractor()
    # Return True si il manque des drones sur au moins un Extractor, False sinon:
    def need_more_drones_on_extractor(self, obs):
        extractors = self.get_my_units_by_type(obs, units.Zerg.Extractor)
        if(len(extractors) > 0):
            for extractor in extractors:
                if(extractor.build_progress < 100):
                    continue
                if(2 - extractor.assigned_harvesters) > 0:
                    return True
        return False

    # Trouve un extractor qui n'a pas 3 drones et le complète:
    # TO DO: ajouter les drones les plus proches
    def put_drones_on_extractor(self, obs):
        extractors = self.get_my_units_by_type(obs, units.Zerg.Extractor)
        if(len(extractors) > 0):
            drones = [drone.tag for drone in self.get_my_units_by_type(
                obs, units.Zerg.Drone) if drone.tag not in self.gas_harvesters]
            for extractor in extractors:
                missing = 2 - extractor.assigned_harvesters
                drones_tag = []
                if missing > 0:
                    if len(drones) > 2:
                        for i in range(missing):
                            drones_tag.append(drones[i])
                            self.gas_harvesters.append(drones[i])
                if drones_tag != []:
                    return actions.RAW_FUNCTIONS.Harvest_Gather_Drone_unit("now", drones_tag, extractor.tag)
        return actions.RAW_FUNCTIONS.no_op()

    # Pour chaque case du SCREEN_FEATURES on appel check_square_to_build_2() jusqu'à trouver un carré valide pour construire.

    def check_square_to_build_1(self, arr, weight):
        for y in range(0, self.SCREEN_DIM_XY-weight):
            for x in range(0, self.SCREEN_DIM_XY-weight):
                if(self.check_square_to_build_2(arr, x, y, weight)):
                    # print("X="+str(x)+"    Y="+str(y))
                    return [x, y]
        return None

    # A partir d'une certaines coordonnée donnée par check_square_to_build_1() on check si chaque case d'un carré de largeur "weight" est valide pour build.

    def check_square_to_build_2(self, arr, x, y, weight):
        for j in range(y, y+weight):
            for i in range(x, x+weight):
                if(arr[j][i] == 0):
                    return False
        return True

    def find_XY_to_build_near_center(self, obs, id):
    # Autour d'un centre, on essaie de trouver des coordonnées pour construire un batiment
    # (dans la version actuelle de code : "centre" == "hatchery")
        # La largeur du carré qu'on cherche = largeur du batiment + 2 (pour avoir un bordure de 1 tout autour)
        weight = techtree[self.current_build_train]["size"] + 2
    # On construit arr[][] qui donne les parties du screen où l'on peut construire
    # Une case de arr[][] vaut 1 ssi: 1/il y a du creep ET 2/il n'y a aucune unité à cet endroit ET 3/le terrain est plat
        unit_type_array = (
            np.array(obs.observation.feature_screen.unit_type) == 0)
        arr = np.multiply(np.multiply(np.array(obs.observation.feature_screen.buildable), np.array(
            obs.observation.feature_screen.creep)), unit_type_array)
        '''for row in arr:
           # print()
            for elmt in row:
               # print(elmt, end=' ')
       # print()'''
    # On cherche un carré assez grand pour recevoir un batiment (+ une bordure autour de se batiment afin de ne jamais être bloqué)
        coord = self.check_square_to_build_1(arr, weight)
        # print(coord)
        # print(self.SCREEN_DIM_XY // 2)
        # Des coordonnées valides on été trouvées :
        if(coord):
            action = info_actions[self.current_build_train]["name"]
            coord[0] += coord[0] + weight
            coord[0] = coord[0] // 2
            coord[0] -= self.SCREEN_DIM_XY // 2
            coord[0] = 2 * coord[0]
            coord[0] += self.camera_coord[0]

            coord[1] += coord[1] + weight
            coord[1] = coord[1] // 2
            coord[1] -= self.SCREEN_DIM_XY // 2
            coord[1] = 2 * coord[1]
            coord[1] += self.camera_coord[1]
            # coord[0] += self.camera_coord[0] - self.SCREEN_DIM_XY//2 #On convertit les coordonnées SCREEN_FEATURES en
            # coord[1] += self.camera_coord[1] - self.SCREEN_DIM_XY//2 #coordonnées RAW
            # print(coord)
            self.center_checked_to_find_XY = []
            # self.build_work_in_progress.append(id)######### ?????????????????????????????????????????????
            drones = self.get_my_units_by_type(obs, units.Zerg.Drone)
            if(len(drones) > 0):
                self.FSM.done()
                # On build le batiment.
                return action('now', drones[0].tag, coord)
            self.FSM.done()
            return actions.RAW_FUNCTIONS.no_op()
        # Pas de coordonnées trouvées sur le centre actuel, on cherche donc un nouveau centre :
        self.FSM.center_wanted()
        return self.find_center_to_build(obs, id)

    # La fonction find_center_to_build() est call lorqu'on veut trouver des coordonnées pour construire un batiment (donc call dans build()).
    # Pour trouver des coordonnées valides on suit plusieurs étapes:
    # - Trouver un centre sur lequel on positionne la caméra
    # - Trouver un carré valide autour du centre pour construire un batiment
    # - Si on ne trouve pas de carré valide on change de centre
    # - Si on a essayé tout les centres disponibles sans trouver de carré valide on crée un nouveau centre
    # (dans la version actuelle de code : "centre" == "hatchery")

    def find_center_to_build(self, obs, id):
        hatcheries = self.get_my_units_by_type(obs, units.Zerg.Hatchery)
        hatcheries_not_checked = [[hatcherie.x, hatcherie.y] for hatcherie in hatcheries
        if [hatcherie.x, hatcherie.y] not in self.center_checked_to_find_XY]
        # Situation où on a essayé tout les centres sans trouver de carré valide pour build.
        # On construit donc un nouveau centre.
        if(len(hatcheries_not_checked) == 0):
            self.center_checked_to_find_XY = []  # Remise à zero des centres checkés
            drones = self.get_my_units_by_type(obs, units.Zerg.Drone)
            test = self.units_in_progress(obs, units.Zerg.Hatchery)
            if(len(drones) > 0 and not self.units_in_progress(obs, units.Zerg.Hatchery)):
                # REMPLACER L'ALEATOIRE PAR
                rnd_x = random.randint(0, self.RAW_RES - 1)
                # DES COORDONNEES HARCODEES ?
                rnd_y = random.randint(0, self.RAW_RES - 1)
                self.FSM.done()
                return actions.RAW_FUNCTIONS.Build_Hatchery_pt('now', drones[0].tag, (rnd_x, rnd_y))
            self.FSM.done()
            return actions.RAW_FUNCTIONS.no_op()
        # Situation où on a encore au moins un centre à checker pour trouver un carré valide pour build.
        # On mémorise ce centre dans self.center_checked_to_find_XY puis positionne la caméra sur ce centre.
        self.center_checked_to_find_XY.append(
            [hatcheries_not_checked[0][0], hatcheries_not_checked[0][1]])
        self.camera_coord = [hatcheries_not_checked[0]
            [0], hatcheries_not_checked[0][1]]
        self.FSM.coord_wanted()
        return actions.RAW_FUNCTIONS.raw_move_camera(self.camera_coord)

    def build(self, obs, id, main_call):
        if(main_call == 0 and self.units_in_progress(obs, id)):
            self.FSM.done()
            return actions.RAW_FUNCTIONS.no_op()

        # Cas de build(Hatchery), on a besoin de coordonnées sans avoir besoin du creep.
        # Donc pas la même logique de coordonnées que les autres batiments.
        if(id == 86):
            drones = self.get_my_units_by_type(obs, units.Zerg.Drone)
            self.FSM.done()
            if(len(drones) > 0 and len(self.expand_coord) > 0):
                coord = self.expand_coord[1]
                return actions.RAW_FUNCTIONS.Build_Hatchery_pt('now', drones[0].tag, coord)
            return actions.RAW_FUNCTIONS.no_op()

        # can_build() return True si on possède tout ce qui est nécessaire à la création de l'unit "id".
        # Si certaines units manquent, can_build return le requierement le plus fondamental.
        # Exemple: Hive needs Infestation_pit,  Infestation_pit needs Lair, Lair needs Spawning_Pool and Hatchery.
        # Si on veut build un Hive, qu'on ne possède pas d'Infestation_pit et pas de Lair mais qu'on possède une Spawning_pool et une Hatchery :
        # can_build(Hive) va return l'id du Lair pour qu'on lance sa construction.
        # Au prochain appel de build(Roach) il manquera un Infestation_pit (si on ne l'a pas construit entre-temps).
        # Dans ce cas can_build(Roach) fera un return de l'id de l'Infestation_pit. Ect
        need = self.can_build(obs, id)
        if(need == True):
            # On accède aux unités à partir desquelles on peut morph/train "id":
            unit_to_morph = self.get_my_units_by_type(
                obs, techtree[id]["morph_from_id"][0])
            # Cas où il est nécessaire de trouver des coordonnées pour construire le batiment (ex: Spawning_pool) :
            if(len(unit_to_morph) > 0 and techtree[id]["needs_coord"]):
                self.current_build_train = id
                self.FSM.center_wanted()
                return actions.RAW_FUNCTIONS.no_op()
            # Cas où il n'est pas nécessaire de trouver des coordonnées pour construire le batiment (ex: Lair) :
            if(len(unit_to_morph) > 0):
                action = info_actions[id]["name"]
                self.FSM.done()
                return action('now', unit_to_morph[0].tag)
        # Cas où on ne possède pas tout le nécessaire pour build "id".
        # On tente de build ce qu'il manque.
        self.current_build_train = need
        return self.build(obs, need, 0)

    # Train

    def train(self, obs, id, main_call):
        # Pour une larva on ne fait rien :
        if(id == 151):
            self.FSM.done()
            return actions.RAW_FUNCTIONS.no_op()
        if(main_call == 0 and self.units_in_progress(obs, id)):
            self.FSM.done()
            return actions.RAW_FUNCTIONS.no_op()
        # can_build() return True si on possède tout ce qui est nécessaire à la création de l'unit "id".
        # Si certaines units manquent, can_build return le requierement le plus fondamental.
        # Exemple: Roach needs Warren_Roach,  Warren_roach needs Spawning_pool, Spawning_pool needs Hatchery.
        # Si on veut train un Roach, qu'on ne possède pas de Warren_roach et pas de Spawning_pool mais qu'on possède une Hatchery :
        # can_build(Roach) va return l'id de la Spawning_pool pour qu'on lance sa construction.
        # Au prochain appel de train(Roach) il manquera une Warren_roach (si on ne l'a pas construit entre-temps).
        # Dans ce cas can_build(Roach) fera un return de l'id de la Warren_roach. Ect
        need = self.can_build(obs, id)
        if(need == True):
            # On accède aux unités à partir desquelles on peut morph/train "id":
            unit_to_morph = self.get_my_units_by_type(
                obs, techtree[id]["morph_from_id"][0])
            if(len(unit_to_morph) > 0):
                action = info_actions[id]["name"]
                self.FSM.done()
                return action("now", unit_to_morph[0])
            self.FSM.done()
            return actions.RAW_FUNCTIONS.no_op()
        # Cas où on ne possède pas tout le nécessaire pour train "id" et que le requierement return par can_build() est un batiment.
        # On tente de build ce qu'il manque.
        elif(techtree[need]["is_building"]):
            self.FSM.building_wanted()
            self.current_build_train = need
            return self.build(obs, need, 0)
        # Cas où on ne possède pas tout le nécessaire pour train "id" et que le requierement return par can_build() n'est pas un batiment.
        # On tente de train ce qu'il manque.
        else:
            self.current_build_train = need
            return self.train(obs, need, 0)
        self.FSM.done()
        return actions.RAW_FUNCTIONS.no_op()

    def ready(self, obs):
        if((obs.observation.player.food_cap - obs.observation.player.food_used) < 10):
            self.FSM.train_wanted()  # Train Overlord
            return self.train(obs, 106, 1)
        if(len(self.get_my_units_by_type(obs, units.Zerg.Drone)) < 16):
            self.FSM.train_wanted()  # Train Drone
            return self.train(obs, 104, 1)
        if(self.need_more_drones_on_extractor(obs)):
            return self.put_drones_on_extractor(obs)
        if(len(self.build_queue) == 0):
            return actions.RAW_FUNCTIONS.no_op()
        if(len(self.build_queue) > 0):
            if(techtree[self.build_queue[0]]["is_building"]):
                self.FSM.building_wanted()
                self.current_build_train = self.build_queue[0]
                return self.build(obs, self.build_queue[0], 1)
            else:
                # print(len(self.get_my_units_by_type(obs, units.Zerg.Larva)))
                self.FSM.train_wanted()
                self.current_build_train = self.build_queue[0]
                return self.train(obs, self.build_queue[0], 1)
        return actions.RAW_FUNCTIONS.no_op()

    def step(self, obs):

        super(ZergAgent, self).step(obs)

        self.step_acc += 1
        self.step_count += 1

        # print(f'steps : {self.step_count}')
        # print(f'rows : {self.step_count // 10}')

        global bot_name
        global bot_race
        


        if obs.first():

            match_stats = ['score','idle_production_time','idle_worker_time','total_value_units','total_value_structures','killed_value_units','killed_value_structures',
                            'collected_minerals','collected_vespene','collection_rate_minerals','collection_rate_vespene','spent_minerals','spent_vespene']

            global df_match_stats
            df_match_stats = pd.DataFrame(columns=match_stats)


        if self.step_count % 10 == 0:
            df_match_stats.loc[self.step_count // 10] = obs.observation['score_cumulative']

        #if self.step_acc % 200 == 0:
        if obs.last():

            

            df_match_stats['number_roaches'] = ROACH_NUMBER
            df_match_stats['result'] = obs.reward
            
            df_match_stats['race'] = bot_name
            df_match_stats['difficulty'] = sys.argv[1]
            

            files = [f for f in os.listdir('stats/matches/') if os.path.isfile(os.path.join('stats/matches/', f))]


            if obs.reward == 1:
                result = 'win'
            elif obs.reward == -1:
                result = 'loss'
            else:
                result = 'draw'


            
            if len(files) > 0:
                filename = str(int(files[-1][:5])+1).zfill(5)
                df_match_stats.to_csv(f'stats/matches/{filename}_{bot_name}_{result}_{sys.argv[1]}_N={str(ROACH_NUMBER).zfill(2)}.csv')
            else:
                df_match_stats.to_csv(f'stats/matches/00001_{bot_name}_{result}_{sys.argv[1]}_N={str(ROACH_NUMBER).zfill(2)}.csv')


            if bot_race == 1:
  
                if obs.reward == 1:
                    df_wins.loc[df_wins.Difficulty == sys.argv[1], 'ProWins'] += 1
                elif obs.reward == -1:
                    df_wins.loc[df_wins.Difficulty == sys.argv[1], 'ProLosses'] += 1
                else:
                    df_wins.loc[df_wins.Difficulty == sys.argv[1], 'ProStalemates'] += 1

            elif bot_race == 2:

                if obs.reward == 1:
                    df_wins.loc[df_wins.Difficulty == sys.argv[1], 'TerWins'] += 1
                elif obs.reward == -1:
                    df_wins.loc[df_wins.Difficulty == sys.argv[1], 'TerLosses'] += 1
                else:
                    df_wins.loc[df_wins.Difficulty == sys.argv[1], 'TerStalemates'] += 1

            else:
  
                if obs.reward == 1:
                    df_wins.loc[df_wins.Difficulty == sys.argv[1], 'ZerWins'] += 1
                elif obs.reward == -1:
                    df_wins.loc[df_wins.Difficulty == sys.argv[1], 'ZerLosses'] += 1
                else:
                    df_wins.loc[df_wins.Difficulty == sys.argv[1], 'ZerStalemates'] += 1

            df_wins.to_csv('stats/df_wins.csv')
            setup_data()

        
    #    if(self.step_acc > 200):
    #        self.step_acc = 0
    #        self.id_attack = (self.id_attack + 1) % 2

        if self.step_acc % 600 == 0:
            self.id_attack = 1
        elif self.step_acc % 200 == 0:
            self.id_attack = 0

        if not(self.FSM.current == "ready"):
            return self.step_FSM(obs)
        return self.step_BO(obs)

    def step_BO(self, obs):
        if(self.BO.current == "phase_0"):
            return self.phase_0(obs)
        if(self.BO.current == "phase_1"):
            return self.phase_1(obs)
        if(self.BO.current == "phase_2"):
            return self.phase_2(obs)
        if(self.BO.current == "phase_3"):
            return self.phase_3(obs)
        if(self.BO.current == "phase_4"):
            return self.phase_4(obs)
        if(self.BO.current == "phase_5"):
            return self.phase_5(obs)
        if(self.BO.current == "phase_6"):
            return self.phase_6(obs)
        if(self.BO.current == "phase_7"):
            return self.phase_7(obs)
        if(self.BO.current == "phase_8"):
            return self.phase_8(obs)
        return actions.RAW_FUNCTIONS.no_op()

    def step_FSM(self, obs):
        '''
        unit_type_array = (
            np.array(obs.observation.feature_screen.unit_type) == 0) * 1
        arr = np.multiply(np.multiply(np.array(obs.observation.feature_screen.buildable),np.array(
            obs.observation.feature_screen.creep)),unit_type_array)
        for row in unit_type_array:
           # print()
            for elmt in row:
               # print(elmt, end=' ')
       # print()

        hatcherie = self.get_my_units_by_type(obs, units.Zerg.Hatchery)[0]
        if(self.centred == 0):
            self.tag = self.get_my_units_by_type(obs, units.Zerg.Drone)[0].tag
            self.drone = self.get_my_units_by_type(obs, units.Zerg.Drone)[0]
            self.centred = 1
            return actions.RAW_FUNCTIONS.raw_move_camera((2*37, 2*38))
       # print(self.drone.x,self.drone.y)
        # return actions.RAW_FUNCTIONS.Build_Hatchery_pt('now', self.tag, (39 + 16, 42 + 0))
        return actions.RAW_FUNCTIONS.Move_pt("now",self.tag, (2*37 + 2* 14, 2*38 + 2* 10))
        '''

        '''
        self.FSM.building_wanted()
        return self.build(obs, self.build_queue[0], 1)
        '''

        '''
        drones = self.get_my_units_by_type(obs,units.Zerg.Drone)
        for drone in drones:
           # print(drone.order_id_0, drone.order_id_1)
        '''

        '''
        # super(ZergAgent, self).step(obs)
        if(self.FSM.current == "start"):
            return self.start(obs)
        if(self.FSM.current == "build_extractor"):
            return self.build_extractor(obs)
        '''
        if(self.FSM.current == "ready"):
            return self.ready(obs)
        if(self.FSM.current == "find_center"):
            return self.find_center_to_build(obs, self.current_build_train)
        if(self.FSM.current == "find_coord"):
            return self.find_XY_to_build_near_center(obs, self.current_build_train)

        return actions.RAW_FUNCTIONS.no_op()





difficulties = {
    'very_easy': sc2_env.Difficulty.very_easy,
    'easy': sc2_env.Difficulty.easy,
    'medium': sc2_env.Difficulty.medium,
    'medium_hard': sc2_env.Difficulty.medium_hard,
    'hard': sc2_env.Difficulty.hard,
    'harder': sc2_env.Difficulty.harder,
    'very_hard': sc2_env.Difficulty.very_hard,
}

enemy_bots = {
    1: sc2_env.Race.protoss, 
    2: sc2_env.Race.terran,
    3: sc2_env.Race.zerg,
}

bot_race = None
bot_name = 'zerg'
df_wins = pd.DataFrame()
df_match_stats = pd.DataFrame()


bot_race = random.randint(1, 3)
if bot_race == 1:
    bot_name = 'protoss'
elif bot_race == 2:
    bot_name = 'terran'
elif bot_race == 3:
    bot_name == 'zerg'

enemy_bot = enemy_bots[bot_race]

print(f'bot race : {bot_race}')
print(f'bot name : {bot_name}')

def setup_data():
    print('call')
    global bot_name
    bot_name = 'zerg'
    global bot_race
    bot_race = random.randint(1, 3)
    if bot_race == 1:
        bot_name = 'protoss'
    elif bot_race == 2:
        bot_name = 'terran'
    elif bot_race == 3:
        bot_name == 'zerg'

    global ROACH_NUMBER
    ROACH_NUMBER = random.randint(3, 16)

    global enemy_bot
    enemy_bot = enemy_bots[bot_race]

#    print(f'bot race : {bot_race}')
#    print(f'bot name : {bot_name}')



ROACH_NUMBER = random.randint(3, 16)



def main(unused_argv):


    difficulty_level = difficulties[sys.argv[1]]

    if os.path.isfile('stats/df_wins.csv'):   
        global df_wins
        df_wins = pd.read_csv('stats/df_wins.csv', index_col=0)
    else:
        difficulties_list = list(difficulties)
        df_wins = pd.DataFrame(columns=['Difficulty', 'TerWins', 'TerLosses', 'TerStalemates', 'ProWins', 'ProLosses', 'ProStalemates', 'ZerWins', 'ZerLosses', 'ZerStalemates'])
        df_wins['Difficulty'] = difficulties_list
        for col in df_wins.columns[1:]:
            df_wins[col].values[:] = 0
        df_wins.to_csv('stats/df_wins.csv')


    agent = ZergAgent()
    agent.reset()

    try:
        while True:

            # print(f'bot race : {bot_race}')

            with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.zerg), 
                        #sc2_env.Bot(enemy_bots[bot_race], difficulties[bot_difficulty])],
                        sc2_env.Bot(enemy_bot, difficulty_level)],   
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=29, minimap=100),
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution = 182,
                    camera_width_world_units = 32
                ),
                
                # disable_fog=True,
                visualize=False) as env:
                    run_loop.run_loop([agent], env, max_frames=1400)
                    # run_loop.run_loop([agent], env)

            # df_wins.to_csv('stats/df_wins.csv')
      
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("\nDifficulty not provided\n")
        sys.exit()
        
    if sys.argv[1] not in list(difficulties):
        print(f"\nDifficulty '{sys.argv[1]}' doesn't exist\n")
        sys.exit()

    if not os.path.exists('stats'):
        os.mkdir('stats')

    if not os.path.exists('stats/matches'):
        os.mkdir('stats/matches')

    # setup_data()
    app.run(main)
            



        
