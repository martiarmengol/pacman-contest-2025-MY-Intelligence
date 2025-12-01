# my_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import random
import util
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='SmartOffensiveAgent', second='SmartDefensiveAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Base Class #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    Reflex Capture Agent base class.
    Contains helper methods for map analysis and successor generation.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.choke_points = []
        self.home_border = []  # ← NUEVO: casillas seguras para "volver a casa"

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.choke_points = self.identify_choke_points(game_state)

        # Cálculo de la frontera de casa (línea de medio mapa de tu lado)
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        mid_x = width // 2

        if self.red:
            border_x = mid_x - 1
        else:
            border_x = mid_x

        self.home_border = [
            (border_x, y)
            for y in range(height)
            if not game_state.has_wall(border_x, y)
        ]

        CaptureAgent.register_initial_state(self, game_state)

    def distance_to_home(self, pos):
        """
        Distancia mínima desde 'pos' a una casilla segura de nuestro lado.
        Si por algún motivo no tenemos home_border, usamos start.
        """
        if not self.home_border:
            return self.get_maze_distance(pos, self.start)
        return min(self.get_maze_distance(pos, b) for b in self.home_border)


    def identify_choke_points(self, game_state):
        width, height = game_state.data.layout.width, game_state.data.layout.height
        choke_points = []
        for x in range(width):
            for y in range(height):
                if game_state.has_wall(x, y): continue
                neighbors = sum([not game_state.has_wall(nx, ny)
                                 for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]])
                if neighbors == 2:
                    choke_points.append((x, y))
        return choke_points

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        return util.Counter()

    def get_weights(self, game_state, action):
        return {}

    def distance_to_home(self, game_state, pos):
        """
        Returns the maze distance from pos to the nearest position on our home border.
        Falls back to the agent start position if the border is blocked.
        """
        if pos is None:
            return 0

        width = game_state.data.layout.width
        height = game_state.data.layout.height

        if self.red:
            home_x = (width // 2) - 1
        else:
            home_x = width // 2

        border_positions = [
            (home_x, y)
            for y in range(height)
            if not game_state.has_wall(home_x, y)
        ]

        if not border_positions:
            if self.start:
                border_positions = [self.start]
            else:
                return 0

        distances = [
            self.get_maze_distance(pos, target)
            for target in border_positions
        ]

        # Filter out impossible paths
        distances = [d for d in distances if d is not None]

        if not distances:
            return 0

        return min(distances)

    def _get_safe_path(self, game_state, target_pos):
        """
        Finds a path to the target using A* search, avoiding enemies.
        Returns the first action of the path, or None if no path exists.
        """
        if not target_pos:
            return None

        my_pos = game_state.get_agent_position(self.index)
        if my_pos == target_pos:
            return None

        # Identify unsafe positions (ghosts)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        unsafe_positions = []
        for enemy in enemies:
            if not enemy.is_pacman and enemy.get_position() and enemy.scared_timer < 5:
                unsafe_positions.append(enemy.get_position())

        # A* Search
        from util import PriorityQueue
        frontier = PriorityQueue()
        frontier.push((my_pos, []), 0)
        visited = set()
        walls = game_state.get_walls()

        while not frontier.is_empty():
            current_pos, path = frontier.pop()

            if current_pos == target_pos:
                return path[0] if path else None

            if current_pos not in visited:
                visited.add(current_pos)
                
                # Check neighbors
                x, y = current_pos
                for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                    dx, dy = 0, 0
                    if action == Directions.NORTH: dy = 1
                    elif action == Directions.SOUTH: dy = -1
                    elif action == Directions.EAST: dx = 1
                    elif action == Directions.WEST: dx = -1
                    
                    next_x, next_y = int(x + dx), int(y + dy)
                    if not walls[next_x][next_y]:
                        next_pos = (next_x, next_y)
                        
                        # Safety Check
                        is_safe = True
                        for unsafe in unsafe_positions:
                            if util.manhattan_distance(next_pos, unsafe) <= 1:
                                is_safe = False
                                break
                        
                        if is_safe:
                            new_cost = len(path) + 1 + util.manhattan_distance(next_pos, target_pos)
                            frontier.push((next_pos, path + [action]), new_cost)
        return None    


##########
# Offensive Agent #
##########

class SmartOffensiveAgent(ReflexCaptureAgent):
    """
    Smart Offensive Agent mejorado:
    - Usa el marcador real del juego (self.get_score).
    - Cambia dinámicamente entre modo 'attack' y 'retreat'.
    - Safety solo actúa cuando hay fantasmas activos visibles.
    - No penaliza choke points si no hay peligro real.
    - Recompensa fuerte por comer comida y por acercarse a comida cercana.
    - Evita bucles con memoria de posiciones.
    - Usa Monte Carlo SOLO cuando está siendo campeado al intentar entrar a territorio enemigo.
    """

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.mode = "attack"  # attack | retreat
        self.visited_positions = util.Counter()

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # 1) Actualizar historial de posiciones con decaimiento
        for pos in list(self.visited_positions.keys()):
            self.visited_positions[pos] *= 0.9
            if self.visited_positions[pos] < 0.1:
                del self.visited_positions[pos]

        if my_state.num_returned > 0:
            self.visited_positions.clear()

        self.visited_positions[my_pos] += 1

        # 2) Analizar enemigos y decidir modo (attack/retreat)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None]
        active_ghosts = [g for g in ghosts if g.scared_timer == 0]
        scared_ghosts = [g for g in ghosts if g.scared_timer > 0]

        close_active_ghost = False
        if active_ghosts and my_state.is_pacman:
            dists = [self.get_maze_distance(my_pos, g.get_position()) for g in active_ghosts]
            close_active_ghost = (min(dists) <= 3)

        time_left = game_state.data.timeleft

        if my_state.num_carrying >= 4 or time_left < 200 or close_active_ghost:
            self.mode = "retreat"
        else:
            self.mode = "attack"

        # 3) Acciones legales
        actions = game_state.get_legal_actions(self.index)

        # 4) Safety (solo cuando hay fantasmas activos visibles)
        safe_actions = self.get_safe_actions(game_state, actions)
        if not safe_actions:
            safe_actions = actions

        # MODO "MONTE CARLO SOLO PARA ENTRAR"
        loop_level = self.visited_positions[my_pos]
        if self.is_frontier_camped(game_state, my_state, my_pos, active_ghosts, loop_level):
            candidate_actions = actions
            return self.monte_carlo_probe(game_state, candidate_actions,
                                           num_rollouts=10, depth=14)

        # Si no hay fantasmas activos y la comida está relativamente cerca,
        # elegimos la acción que minimiza la distancia a la comida tras movernos.
        food_list = self.get_food(game_state).as_list()

        min_food_dist = None
        if food_list:
            min_food_dist = min(self.get_maze_distance(my_pos, f) for f in food_list)

        if not active_ghosts and min_food_dist is not None and min_food_dist <= 4:
            # No queremos considerar STOP en el modo greedy,
            # salvo que sea la única acción posible.
            candidate_actions = [a for a in safe_actions if a != Directions.STOP]
            if not candidate_actions:
                candidate_actions = safe_actions  # solo STOP disponible

            def dist_to_food_after(a):
                succ = self.get_successor(game_state, a)
                succ_pos = succ.get_agent_position(self.index)

                # Usamos SIEMPRE la lista de comida del estado ACTUAL,
                # para que comer un punto cercano se considere "acercarse" a él.
                return min(self.get_maze_distance(succ_pos, f) for f in food_list)

            best_dist = min(dist_to_food_after(a) for a in candidate_actions)
            best_food_actions = [a for a in candidate_actions if dist_to_food_after(a) == best_dist]

            return random.choice(best_food_actions)


        # 5) ROMPE–BUCLES FUERTE (para otros casos)
        if loop_level >= 2:
            candidate_actions = safe_actions if safe_actions else actions

            def successor_visits(a):
                succ = self.get_successor(game_state, a)
                succ_pos = succ.get_agent_position(self.index)
                return self.visited_positions[succ_pos]

            best_loop_action = min(candidate_actions, key=successor_visits)
            return best_loop_action

        # 6) Evaluar normalmente si no estamos en bucle
        values = [self.evaluate(game_state, a) for a in safe_actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(safe_actions, values) if v == max_value]

        if Directions.STOP in best_actions and len(best_actions) > 1:
            best_actions.remove(Directions.STOP)

        return random.choice(best_actions)

    # Helpers Monte Carlo SOLO PARA ENTRAR

    def is_frontier_camped(self, game_state, my_state, my_pos, active_ghosts, loop_level):
        """
        Detecta la situación: quiero entrar a territorio enemigo pero me campean en la frontera.

        Condiciones:
        - Soy FANTASMA (aún en mi lado) -> my_state.is_pacman == False
        - Estoy en modo ataque (quiero ir a comer comida).
        - Hay fantasmas activos visibles.
        - Llevo un mínimo de visitas en esta casilla (bucle local).
        - Estoy cerca de la frontera.
        - Algún fantasma activo está relativamente cerca.
        """
        # Solo nos interesa este Monte Carlo cuando AÚN NO soy Pacman
        if my_state.is_pacman:
            return False

        # Solo tiene sentido en modo ataque (no volviendo a casa)
        if self.mode != "attack":
            return False

        if not active_ghosts:
            return False

        if loop_level < 1:
            return False

        # Frontera vertical aproximada
        width = game_state.data.layout.width
        if self.red:
            frontier_x = (width // 2) - 1
        else:
            frontier_x = (width // 2)

        # ¿Estoy razonablemente cerca de la frontera?
        if abs(my_pos[0] - frontier_x) > 3:
            return False

        # Distancia al fantasma activo más cercano
        dists = [
            self.get_maze_distance(my_pos, g.get_position())
            for g in active_ghosts
            if g.get_position() is not None
        ]
        if not dists:
            return False

        min_dist = min(dists)

        # “Cerca”: umbral ajustable
        return min_dist <= 5

    def monte_carlo_probe(self, game_state, candidate_actions,
                           num_rollouts=10, depth=14):
        """
        Monte Carlo simple:
        - Para cada acción candidata, simula varias trayectorias aleatorias
          (sin STOP y evitando ir siempre marcha atrás).
        - Evalúa el estado final con self.evaluate.
        - Escoge la acción con mejor valor medio.
        """
        # Evitar STOP como acción inicial si hay alternativas
        filtered_candidates = [a for a in candidate_actions if a != Directions.STOP]
        if filtered_candidates:
            candidate_actions = filtered_candidates

        best_action = None
        best_value = -float('inf')

        for a in candidate_actions:
            total = 0.0
            for _ in range(num_rollouts):
                state = game_state.generate_successor(self.index, a)
                steps = depth
                while steps > 0:
                    legal = state.get_legal_actions(self.index)
                    if not legal:
                        break
                    # No quedarnos parados en el rollout
                    if Directions.STOP in legal and len(legal) > 1:
                        legal.remove(Directions.STOP)
                    # Evitar ir siempre marcha atrás
                    cur_dir = state.get_agent_state(self.index).configuration.direction
                    rev = Directions.REVERSE[cur_dir]
                    if rev in legal and len(legal) > 1:
                        legal.remove(rev)

                    next_a = random.choice(legal)
                    state = state.generate_successor(self.index, next_a)
                    steps -= 1

                # Usamos evaluate sobre el estado final
                total += self.evaluate(state, Directions.STOP)

            avg_value = total / float(num_rollouts)
            if avg_value > best_value:
                best_value = avg_value
                best_action = a

        # Por seguridad
        if best_action is None:
            return random.choice(candidate_actions)

        return best_action

    def get_safe_actions(self, game_state, actions):
        """
        Capa de seguridad:
        - NO hace nada si no hay fantasmas activos visibles.
        - Si los hay, evita casillas ADYACENTES al fantasma.
        - Si todas son malas, escoge la que maximiza la distancia al fantasma más cercano.
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        defenders = [a for a in enemies
                     if not a.is_pacman and a.get_position() is not None and a.scared_timer == 0]

        # Si no hay fantasmas activos visibles, no filtramos nada
        if len(defenders) == 0:
            return actions

        my_pos = game_state.get_agent_position(self.index)
        safe_actions = []

        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_position(self.index)

            is_safe = True
            for defender in defenders:
                # Solo consideramos peligrosa una casilla adyacente al fantasma
                if self.get_maze_distance(successor_pos, defender.get_position()) <= 1:
                    is_safe = False
                    break
            if is_safe:
                safe_actions.append(action)

        if not safe_actions:
            # Elegimos la "menos mala": maximiza distancia al fantasma más cercano
            closest_defender = min(
                defenders,
                key=lambda d: self.get_maze_distance(my_pos, d.get_position())
            )
            best_panic_action = max(
                actions,
                key=lambda a: self.get_maze_distance(
                    self.get_successor(game_state, a).get_agent_position(self.index),
                    closest_defender.get_position()
                )
            )
            return [best_panic_action]

        return safe_actions

    def get_features(self, game_state, action):
        features = util.Counter()

        successor = self.get_successor(game_state, action)
        my_state_succ = successor.get_agent_state(self.index)
        my_pos_succ = my_state_succ.get_position()

        my_state_curr = game_state.get_agent_state(self.index)

        # Marcador real del juego
        features['successor_score'] = self.get_score(successor)

        # Comida enemiga
        food_list_succ = self.get_food(successor).as_list()
        features['distance_to_food'] = 0
        features['remaining_food'] = 0
        features['adjacent_food'] = 0

        if food_list_succ:
            dists_food = [self.get_maze_distance(my_pos_succ, f) for f in food_list_succ]
            features['distance_to_food'] = min(dists_food)
            features['remaining_food'] = len(food_list_succ)

            # Bonus por estar a un solo paso de alguna comida
            for f, d in zip(food_list_succ, dists_food):
                if d == 1:
                    features['adjacent_food'] = 1
                    break

        # BONUS directo por comer comida
        if my_state_succ.num_carrying > my_state_curr.num_carrying:
            features['eat_food'] = my_state_succ.num_carrying - my_state_curr.num_carrying
        else:
            features['eat_food'] = 0

        # Volver a casa cuando llevamos comida
        if my_state_succ.num_carrying > 0:
            features['distance_to_home'] = self.distance_to_home(successor, my_pos_succ)
        else:
            features['distance_to_home'] = 0

        # Cápsulas enemigas
        capsules_succ = self.get_capsules(successor)
        if capsules_succ:
            features['distance_to_capsule'] = min(
                self.get_maze_distance(my_pos_succ, c) for c in capsules_succ
            )
        else:
            features['distance_to_capsule'] = 0

        # Bucles (penalizar posiciones visitadas)
        visited_count = self.visited_positions.get(my_pos_succ, 0.0)
        features['visited_penalty'] = visited_count

        # Relación con fantasmas (activos y asustados)
        enemies_succ = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts_succ = [e for e in enemies_succ if not e.is_pacman and e.get_position() is not None]
        active_ghosts = [g for g in ghosts_succ if g.scared_timer == 0]
        scared_ghosts = [g for g in ghosts_succ if g.scared_timer > 0]

        if active_ghosts:
            dists_active = [self.get_maze_distance(my_pos_succ, g.get_position()) for g in active_ghosts]
            features['active_ghost_distance'] = min(dists_active)
        else:
            features['active_ghost_distance'] = 0

        if scared_ghosts:
            dists_scared = [self.get_maze_distance(my_pos_succ, g.get_position()) for g in scared_ghosts]
            features['scared_ghost_distance'] = min(dists_scared)
        else:
            features['scared_ghost_distance'] = 0

        # Choke points: solo problema si hay fantasmas activos cercanos
        features['in_choke'] = 0
        if my_pos_succ in self.choke_points and active_ghosts:
            dists_active = [self.get_maze_distance(my_pos_succ, g.get_position()) for g in active_ghosts]
            if dists_active and min(dists_active) <= 8:
                features['in_choke'] = 1

        # Penalizar quedarse quieto
        features['stop'] = 1 if action == Directions.STOP else 0

        return features

    def get_weights(self, game_state, action):
        """
        Pesos dinámicos según el modo (attack / retreat) y la situación.
        - Cápsulas solo pesan mucho si hay fantasmas activos cerca.
        - Comer comida tiene un premio muy fuerte.
        - Choke points solo importan cuando hay peligro.
        """
        successor = self.get_successor(game_state, action)
        my_state_succ = successor.get_agent_state(self.index)
        my_pos_succ = my_state_succ.get_position()
        # Distancia a la comida EN get_weights (para reglas especiales)
        foods_succ = self.get_food(successor).as_list()
        dist_food_succ = None
        if foods_succ:
            dist_food_succ = min(self.get_maze_distance(my_pos_succ, f) for f in foods_succ)

        time_left = game_state.data.timeleft

        enemies_succ = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts_succ = [e for e in enemies_succ if not e.is_pacman and e.get_position() is not None]
        active_ghosts = [g for g in ghosts_succ if g.scared_timer == 0]
        scared_ghosts = [g for g in ghosts_succ if g.scared_timer > 0]

        min_active_dist = None
        if active_ghosts:
            dists_active = [self.get_maze_distance(my_pos_succ, g.get_position()) for g in active_ghosts]
            min_active_dist = min(dists_active)

        # Recalcular distancia a cápsula aquí:
        capsules_succ = self.get_capsules(successor)
        if capsules_succ:
            dist_capsule = min(self.get_maze_distance(my_pos_succ, c) for c in capsules_succ)
        else:
            dist_capsule = None

        # Peso base para cápsulas
        capsule_weight = -2

        # Si la cápsula está CERCA (oportunidad clara)
        if dist_capsule is not None and dist_capsule <= 5:
            capsule_weight = -5  # ve a por ella agresivamente

        # Si hay fantasmas activos visibles → comportamiento defensivo clásico
        if min_active_dist is not None:
            if min_active_dist <= 5:
                capsule_weight = -12     # riesgo alto → prioridad máxima
            elif min_active_dist <= 8:
                capsule_weight = -5     # riesgo medio
            else:
                capsule_weight = -3     # riesgo bajo pero presente

        # Si los fantasmas visibles están asustados → baja prioridad
        if active_ghosts == [] and scared_ghosts:
            capsule_weight *= 0.3       # no desperdiciar cápsula

        # Pesos base para modos ataque y retirada

        # MODO ATAQUE: priorizar MUY fuerte ir hacia comida y comerla.
        weights_attack = {
            'successor_score': 50,
            'distance_to_food': -12,
            'remaining_food': -0.5,
            'eat_food': 60,          
            'adjacent_food': 40,     
            'distance_to_home': 0,
            'distance_to_capsule': capsule_weight,
            'visited_penalty': -4,
            'active_ghost_distance': 0,   
            'scared_ghost_distance': 0.5,
            'in_choke': -1.5,
            'stop': -200,
        }

        # MODO RETIRADA: importa llegar a casa y no morir.
        weights_retreat = {
            'successor_score': 220,
            'distance_to_food': -0.5,
            'remaining_food': 0,
            'eat_food': 5,
            'adjacent_food': 5,
            'distance_to_home': -12,
            'distance_to_capsule': capsule_weight,
            'visited_penalty': -5,
            'active_ghost_distance': 5,
            'scared_ghost_distance': 0.3,
            'in_choke': -6,
            'stop': -40,
        }

        # Ajustes según modo, carga y tiempo 
        if self.mode == "attack":
            w = weights_attack

            # Solo empezamos a valorar volver a casa si vamos cargados y hay peligro
            if active_ghosts and my_state_succ.num_carrying >= 3 and min_active_dist is not None:
                if min_active_dist <= 8:
                    w['distance_to_home'] = -3
                else:
                    w['distance_to_home'] = 0
            else:
                w['distance_to_home'] = 0

            if time_left < 200:
                w['distance_to_home'] = -8
                w['successor_score'] = 140
        else:
            w = weights_retreat
            if my_state_succ.num_carrying > 5 or time_left < 150:
                w['distance_to_home'] = -12
                w['successor_score'] = 250

        # Si no hay fantasmas activos y la comida está muy cerca,
        # bajamos mucho la penalización por casillas visitadas para que ENTRE al pasillo.
        if not active_ghosts and dist_food_succ is not None and dist_food_succ <= 2:
            w['visited_penalty'] = 0   # o -1 si quieres un pelín de castigo

        return w


##########
# Defensive Agent #
##########

class SmartDefensiveAgent(ReflexCaptureAgent):
    """
    A defensive agent that patrols and intercepts invaders.
    
    Strategy:
    1. Intercept: If an invader is seen, use Minimax to predict their escape and cut them off.
    2. Food Memory: Track eaten food and patrol that location even when invader disappears.
    3. Patrol: If no enemies visible and no food eaten, patrol border near food clusters.
    4. Capsule Defense: Protect capsules when invaders are nearby.
    5. Escape Route Blocking: Position to cut off invader's path home.
    6. Approximate Hunt: When invader disappears, approximate where it could have moved.

    In short: we guard important tiles (food, capsules, border) and try to stand
    between invading Pacmen and the way back to their home side.
    """
    
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.previous_food = []
        self.target = None
        self.counter = 0
        self.patrol_points = []
        self.last_seen_invader_pos = None
        self.turns_since_seen = 0
        self.enemy_border = []  # Enemy's home border (their escape route)
        self.food_clusters = []  # High-density food areas to patrol
    
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.previous_food = self.get_food_you_are_defending(game_state).as_list()
        self._set_patrol_points(game_state)
        self._compute_enemy_border(game_state)
        self._compute_food_clusters(game_state)
    
    def choose_action(self, game_state):
        """
        Top–level decision logic for the defender.

        High–level order:
          1) If scared, run away from visible invaders.
          2) Otherwise, if we see invaders, try to intercept them.
          3) If food was just eaten, investigate that area.
          4) If invader was near a capsule, defend that capsule.
          5) If invader disappeared recently, approximate its new position.
          6) If nothing else, patrol important defensive areas.
        """
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        
        # 0) Clear target if we already reached it
        if my_pos == self.target:
            self.target = None
        
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position()]
        
        # Track invader positions for prediction when they disappear
        if invaders:
            self.last_seen_invader_pos = invaders[0].get_position()
            self.turns_since_seen = 0
        else:
            self.turns_since_seen += 1
        
        # We sometimes become "scared" (after enemy eats a capsule) and must avoid invaders
        scared_timer = my_state.scared_timer
        
        # 1) If scared, avoid invaders and just patrol safely
        if scared_timer > 0 and invaders:
            return self._evade_and_patrol(game_state, invaders)
        
        # 2) If we see invaders, try to intercept them (chasing the one carrying the most food)
        if invaders:
            # Prioritize invader carrying more food
            priority_invader = max(invaders, 
                key=lambda a: (a.num_carrying * 100 - self.get_maze_distance(my_pos, a.get_position())))
            
            # Get the invader's agent index
            invader_index = None
            for i in self.get_opponents(game_state):
                if game_state.get_agent_state(i).get_position() == priority_invader.get_position():
                    invader_index = i
                    break
            
            if invader_index is not None:
                invader_pos = priority_invader.get_position()
                invader_carrying = priority_invader.num_carrying
                
                # If invader is carrying food, try to intercept their escape route
                if invader_carrying > 0:
                    intercept_pos = self._get_intercept_position(game_state, invader_pos)
                    if intercept_pos:
                        my_dist_to_intercept = self.get_maze_distance(my_pos, intercept_pos)
                        invader_dist_to_intercept = self.get_maze_distance(invader_pos, intercept_pos)
                        
                        # If we can beat them to the intercept point, go there
                        if my_dist_to_intercept < invader_dist_to_intercept:
                            self.target = intercept_pos
                            return self._move_toward_target(game_state, intercept_pos)
                
                # Use Minimax to predict best interception move
                _, best_action = self.minimax(game_state, invader_index, depth=3, 
                                            alpha=-float('inf'), beta=float('inf'), 
                                            is_maximizing=True)
                if best_action:
                    return best_action
        
        # 3) Food Memory: if food disappeared, an invader must have eaten it nearby
        current_food = self.get_food_you_are_defending(game_state).as_list()
        
        if len(current_food) < len(self.previous_food):
            # Food was eaten! Find which one and predict invader's next move
            eaten_food = set(self.previous_food) - set(current_food)
            if eaten_food:
                eaten_pos = eaten_food.pop()
                # Predict where invader is heading based on nearby food
                predicted_target = self._predict_invader_target(game_state, eaten_pos, current_food)
                self.target = predicted_target if predicted_target else eaten_pos
        
        # Update food memory for next turn
        self.previous_food = current_food
        
        # 4) Capsule defense: protect our defensive capsules if an invader was seen nearby
        capsules = self.get_capsules_you_are_defending(game_state)
        if capsules and self.last_seen_invader_pos and self.turns_since_seen < 5:
            closest_capsule = min(capsules, 
                key=lambda c: self.get_maze_distance(self.last_seen_invader_pos, c))
            capsule_dist = self.get_maze_distance(self.last_seen_invader_pos, closest_capsule)
            if capsule_dist < 6:  # Invader was close to capsule
                self.target = closest_capsule
        
        # 5) If invader disappeared recently, approximate where it may have moved
        if not invaders and self.turns_since_seen < 8:
            estimated_pos = self._estimate_invader_position(game_state)
            if estimated_pos:
                self.target = estimated_pos
        
        # 6) If we still have no concrete target, just patrol strategically
        if self.target is None:
            self.target = self._get_best_patrol_target(game_state, current_food, capsules)
        
        # Move toward target
        return self._move_toward_target(game_state, self.target)
    
    def _evade_and_patrol(self, game_state, invaders):
        """
        When we are scared:
        - Prefer actions that keep at least 3 tiles away from any invader.
        - Among safe actions, still try to hover near our patrol points.
        """
        my_pos = game_state.get_agent_position(self.index)
        actions = game_state.get_legal_actions(self.index)
        
        # Filter actions that keep distance from invaders
        safe_actions = []
        for action in actions:
            if action == Directions.STOP:
                continue
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_position(self.index)
            
            min_invader_dist = min(self.get_maze_distance(new_pos, inv.get_position()) 
                                   for inv in invaders)
            if min_invader_dist >= 3:
                safe_actions.append(action)
        
        if safe_actions:
            # Among safe actions, pick one that moves toward patrol point
            if self.patrol_points:
                return min(safe_actions, key=lambda a: 
                    self.get_maze_distance(
                        self.get_successor(game_state, a).get_agent_position(self.index),
                        self.patrol_points[0]))
            return random.choice(safe_actions)
        
        # No safe actions - maximize distance from closest invader
        closest_invader = min(invaders, 
            key=lambda a: self.get_maze_distance(my_pos, a.get_position()))
        return max(actions, key=lambda a: 
            self.get_maze_distance(
                self.get_successor(game_state, a).get_agent_position(self.index),
                closest_invader.get_position()))
    
    def _get_intercept_position(self, game_state, invader_pos):
        """
        Find a good interception tile between the invader and its escape border.

        Preference:
          - A choke point that lies on the path from invader to its closest border tile.
          - If none is suitable, fall back to that closest border tile itself.
        """
        if not self.enemy_border:
            return None
        
        # Find the border position the invader is closest to
        closest_escape = min(self.enemy_border, 
            key=lambda b: self.get_maze_distance(invader_pos, b))
        
        # Find a choke point between invader and their escape
        my_pos = game_state.get_agent_position(self.index)
        
        # Check our choke points that are between invader and escape
        for choke in self.choke_points:
            invader_to_choke = self.get_maze_distance(invader_pos, choke)
            choke_to_escape = self.get_maze_distance(choke, closest_escape)
            invader_to_escape = self.get_maze_distance(invader_pos, closest_escape)
            
            # Choke is on the path if distances roughly add up
            if abs((invader_to_choke + choke_to_escape) - invader_to_escape) <= 2:
                my_dist = self.get_maze_distance(my_pos, choke)
                if my_dist < invader_to_choke:  # We can get there first
                    return choke
        
        return closest_escape
    
    def _predict_invader_target(self, game_state, last_eaten_pos, remaining_food):
        """
        Predict where an invisible invader is heading after eating at last_eaten_pos.

        Intuition:
          - If there is nearby food, assume it goes to the closest of those.
          - Otherwise, assume it is trying to run home towards its border.
        """
        if not remaining_food:
            return last_eaten_pos
        
        # Find nearby food from last eaten position
        nearby_food = [f for f in remaining_food 
                       if self.get_maze_distance(last_eaten_pos, f) <= 3]
        
        if nearby_food:
            # Invader likely going to closest nearby food
            return min(nearby_food, key=lambda f: self.get_maze_distance(last_eaten_pos, f))
        
        # Otherwise predict they're heading home with food
        if self.enemy_border:
            return min(self.enemy_border, 
                key=lambda b: self.get_maze_distance(last_eaten_pos, b))
        
        return last_eaten_pos
    
    def _estimate_invader_position(self, game_state):
        """
        Approximate where a previously-seen invader could be now.

        We do a small square search around the last seen position, restricted to:
        - in–bounds tiles,
        - non–walls,
        - tiles on our side of the map.

        Among those, we simply prefer positions closest (in grid steps) to the last seen point.
        """
        if not self.last_seen_invader_pos:
            return None
        
        # Simple estimation: invader moves ~1 cell per turn
        # Search area expands with turns since seen
        search_radius = min(self.turns_since_seen + 1, 5)
        
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        mid_x = width // 2
        
        best_estimate = None
        best_score = float('inf')
        
        # Check positions near last seen location
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                test_x = int(self.last_seen_invader_pos[0] + dx)
                test_y = int(self.last_seen_invader_pos[1] + dy)

                # Skip out-of-bounds positions
                if test_x < 0 or test_x >= width or test_y < 0 or test_y >= height:
                    continue

                test_pos = (test_x, test_y)

                # Skip walls
                if game_state.has_wall(test_x, test_y):
                    continue
                
                # Check if this position is on our side (where invader would be)
                if self.red and test_pos[0] >= mid_x:
                    continue
                if not self.red and test_pos[0] < mid_x:
                    continue
                
                score = abs(dx) + abs(dy)  # Prefer positions closer to last seen
                
                if score < best_score:
                    best_score = score
                    best_estimate = test_pos
        
        return best_estimate
    
    def _get_best_patrol_target(self, game_state, current_food, capsules):
        """
        Decide which defensive tile to patrol when we have no concrete invader target.

        Priority:
          1) Directly guard the last few remaining food / capsules.
          2) Sometimes sit near capsules as a deterrent.
          3) Patrol near precomputed food clusters.
          4) Fall back to generic border patrol points.
        """
        my_pos = game_state.get_agent_position(self.index)
        
        # If very few food left, guard them directly
        if current_food and len(current_food) <= 4:
            high_priority = current_food + capsules
            if high_priority:
                return random.choice(high_priority)
        
        # Guard capsules if they exist
        if capsules:
            # 30% chance to patrol near capsule
            if random.random() < 0.3:
                return min(capsules, key=lambda c: self.get_maze_distance(my_pos, c))
        
        # Use precomputed food clusters for strategic patrol
        if self.food_clusters:
            # Patrol cluster we're furthest from (to cover territory)
            return max(self.food_clusters, 
                key=lambda c: self.get_maze_distance(my_pos, c))
        
        # Fallback to patrol points
        if self.patrol_points:
            return random.choice(self.patrol_points)
        
        return self.start
    
    def _move_toward_target(self, game_state, target):
        """Move toward target while staying on defense."""
        if not target:
            return random.choice(game_state.get_legal_actions(self.index))
        
        defensive_actions = self._get_defensive_moves(game_state)
        best_action = self._choose_defensive_action(game_state, defensive_actions, target)
        
        if best_action:
            return best_action
        
        action = self._get_safe_path(game_state, target)
        if action:
            return action
        
        return random.choice(game_state.get_legal_actions(self.index))

    def _get_defensive_moves(self, game_state):
        """
        Candidate defensive moves that keep us on the ghost side and avoid
        STOP/reverse unless we are stuck.
        """
        legal = game_state.get_legal_actions(self.index)
        candidate_actions = [a for a in legal if a != Directions.STOP]
        reverse_dir = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        agent_actions = []

        for action in candidate_actions:
            successor = self.get_successor(game_state, action)
            if not successor.get_agent_state(self.index).is_pacman:
                agent_actions.append(action)

        if agent_actions:
            self.counter += 1
        else:
            self.counter = 0

        if reverse_dir not in agent_actions and (self.counter == 0 or self.counter > 4):
            agent_actions.append(reverse_dir)

        return agent_actions

    def _choose_defensive_action(self, game_state, actions, target):
        """
        Pick the move that minimizes the maze distance to the current target.
        """
        if not actions or not target:
            return None

        distances = []
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            next_pos = successor.get_agent_position(self.index)
            distances.append(self.get_maze_distance(next_pos, target))

        if not distances:
            return None

        best_distance = min(distances)
        best_actions = [a for a, d in zip(actions, distances) if d == best_distance]
        return random.choice(best_actions)

    def _set_patrol_points(self, game_state):
        """
        Precompute patrol positions along the central border column,
        weighted by proximity to food clusters.
        """
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        x = (width - 2) // 2
        if not self.red:
            x += 1

        points = [
            (x, y)
            for y in range(1, height - 1)
            if not game_state.has_wall(x, y)
        ]

        # Keep interior positions so patrol stays near the gap
        while len(points) > 2:
            points = points[1:-1]

        self.patrol_points = points if points else [self.start]
    
    def _compute_enemy_border(self, game_state):
        """Compute the enemy's home border (their escape route)."""
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        
        # Enemy border is on the opposite side
        if self.red:
            border_x = width // 2  # Enemy is on right side
        else:
            border_x = width // 2 - 1  # Enemy is on left side
        
        self.enemy_border = [
            (border_x, y)
            for y in range(height)
            if not game_state.has_wall(border_x, y)
        ]
    
    def _compute_food_clusters(self, game_state):
        """
        Identify clusters of food to patrol strategically.
        Uses simple clustering based on proximity.
        """
        food_list = self.get_food_you_are_defending(game_state).as_list()
        if not food_list:
            self.food_clusters = []
            return
        
        # Simple K-means-like clustering with 2-3 clusters
        # Find centroid positions that represent high-density areas
        clusters = []
        remaining_food = list(food_list)
        
        while remaining_food and len(clusters) < 3:
            # Pick the food item furthest from existing clusters
            if not clusters:
                # Start with a random food
                seed = remaining_food[0]
            else:
                seed = max(remaining_food, 
                    key=lambda f: min(self.get_maze_distance(f, c) for c in clusters))
            
            # Find all food within radius 4 of seed
            cluster_food = [f for f in remaining_food 
                          if self.get_maze_distance(f, seed) <= 4]
            
            if cluster_food:
                # Compute centroid (average position)
                avg_x = sum(f[0] for f in cluster_food) / len(cluster_food)
                avg_y = sum(f[1] for f in cluster_food) / len(cluster_food)
                
                # Find nearest valid position to centroid
                centroid = min(cluster_food, 
                    key=lambda f: abs(f[0] - avg_x) + abs(f[1] - avg_y))
                clusters.append(centroid)
                
                # Remove clustered food
                for f in cluster_food:
                    remaining_food.remove(f)
        
        self.food_clusters = clusters

    def minimax(self, game_state, invader_index, depth, alpha, beta, is_maximizing):
        """
        Minimax with alpha–beta pruning to simulate defender vs. one invader.
        
        We alternate turns:
          - Maximizing player: this defender, trying to catch/block the invader.
          - Minimizing player: the invader, trying to escape or stay away.

        Returns: (best_value, best_action) from the current state for the defender.
        """
        # Base case: depth limit reached
        if depth == 0:
            return self.evaluate_defensive_state(game_state, invader_index), None
        
        # Check if invader was caught or escaped
        invader_pos = game_state.get_agent_position(invader_index)
        if not invader_pos:  # Invader returned home (escaped)
            invader_state = game_state.get_agent_state(invader_index)
            # Penalty based on how much food they escaped with
            return -1000 - invader_state.num_returned * 50, None
        
        my_pos = game_state.get_agent_position(self.index)
        if my_pos == invader_pos:  # Caught!
            invader_state = game_state.get_agent_state(invader_index)
            # Bonus for catching invader with food
            return 1000 + invader_state.num_carrying * 100, None
        
        if is_maximizing:
            # Our turn (defender): maximize value (minimize distance to invader)
            max_value = -float('inf')
            best_action = None
            
            actions = game_state.get_legal_actions(self.index)
            # Filter out STOP unless it's the only option
            non_stop_actions = [a for a in actions if a != Directions.STOP]
            if non_stop_actions:
                actions = non_stop_actions
            
            # Move ordering: prioritize moving toward invader and blocking escape
            actions = sorted(actions, key=lambda a: -self._action_value(game_state, a, invader_pos, invader_index))
            
            for action in actions:
                successor = self.get_successor(game_state, action)
                value, _ = self.minimax(successor, invader_index, depth - 1, alpha, beta, False)
                
                if value > max_value:
                    max_value = value
                    best_action = action
                
                alpha = max(alpha, value)
                if beta <= alpha:
                    break  # Beta cutoff
            
            return max_value, best_action
        else:
            # Invader's turn: minimize value (maximize distance from us / escape)
            min_value = float('inf')
            
            invader_actions = game_state.get_legal_actions(invader_index)
            # Sort by escape potential (prefer moves toward their home)
            invader_actions = sorted(invader_actions, 
                key=lambda a: self._invader_action_priority(game_state, a, invader_index))
            
            for action in invader_actions:
                try:
                    successor = game_state.generate_successor(invader_index, action)
                    value, _ = self.minimax(successor, invader_index, depth - 1, alpha, beta, True)
                    
                    min_value = min(min_value, value)
                    beta = min(beta, value)
                    if beta <= alpha:
                        break  # Alpha cutoff
                except:
                    continue  # Skip invalid moves
            
            return min_value, None
    
    def _action_value(self, game_state, action, invader_pos, invader_index):
        """
        Heuristic score for defender actions when ordering moves in minimax.

        Higher is better for us:
          - Strongly rewards moving closer to the invader.
          - Extra bonus when we move in front of the invader's escape route.
          - Small bonus for staying on our defensive side (not becoming Pacman).
        """
        successor = self.get_successor(game_state, action)
        new_pos = successor.get_agent_position(self.index)
        
        # Base score: distance to invader (negative because closer = better)
        dist_to_invader = self.get_maze_distance(new_pos, invader_pos)
        score = -dist_to_invader * 10
        
        # Bonus for blocking escape route
        if self.enemy_border:
            invader_state = game_state.get_agent_state(invader_index)
            if invader_state.num_carrying > 0:
                # Find invader's closest escape
                invader_escape = min(self.enemy_border, 
                    key=lambda b: self.get_maze_distance(invader_pos, b))
                
                # Check if we're getting between invader and escape
                my_dist_to_escape = self.get_maze_distance(new_pos, invader_escape)
                invader_dist_to_escape = self.get_maze_distance(invader_pos, invader_escape)
                
                if my_dist_to_escape < invader_dist_to_escape:
                    score += 50  # Big bonus for blocking escape
        
        # Bonus for staying on defense (not crossing border)
        if not successor.get_agent_state(self.index).is_pacman:
            score += 20
        
        return score
    
    def _invader_action_priority(self, game_state, action, invader_index):
        """
        Heuristic for ordering invader moves in minimax (lower = more attractive for invader).

        Intuition:
          - If invader is carrying food, it prefers moves that reduce distance to its border.
          - In all cases, it prefers staying further away from our defender.
        """
        try:
            successor = game_state.generate_successor(invader_index, action)
            invader_pos = successor.get_agent_position(invader_index)
            invader_state = successor.get_agent_state(invader_index)
            
            if not invader_pos:
                return float('inf')
            
            my_pos = game_state.get_agent_position(self.index)
            
            # Distance from defender (prefer far)
            dist_from_me = self.get_maze_distance(invader_pos, my_pos)
            
            # Distance to escape (prefer close if carrying food)
            escape_priority = 0
            if self.enemy_border and invader_state.num_carrying > 0:
                dist_to_escape = min(self.get_maze_distance(invader_pos, b) 
                                    for b in self.enemy_border)
                escape_priority = dist_to_escape * 2  # Weight escape higher
            
            # Combine: lower = better for invader (escape + avoid defender)
            return escape_priority - dist_from_me
        except:
            return float('inf')

    def evaluate_defensive_state(self, game_state, invader_index):
        """
        Static evaluation for minimax leaves: “how good is this position for the defender?”

        Main factors:
          - Closer to invader is better.
          - Invaders holding more food are more valuable to catch.
          - Invaders far from their border (escape) are good for us.
          - Being on escape chokepoints is strongly rewarded.
          - Crossing into enemy territory as a defender is discouraged.
        """
        my_pos = game_state.get_agent_position(self.index)
        invader_pos = game_state.get_agent_position(invader_index)
        invader_state = game_state.get_agent_state(invader_index)
        
        if not invader_pos:  # Invader escaped
            return -1000 - invader_state.num_returned * 50
        
        score = 0
        
        # 1. Distance to invader (closer = better)
        distance = self.get_maze_distance(my_pos, invader_pos)
        score -= distance * 10
        
        # 2. Bonus for invader carrying food (more valuable to catch)
        if invader_state.num_carrying > 0:
            score += invader_state.num_carrying * 25
        
        # 3. Bonus for invader being far from their escape
        if self.enemy_border:
            invader_dist_to_escape = min(self.get_maze_distance(invader_pos, b) 
                                         for b in self.enemy_border)
            score += invader_dist_to_escape * 5  # Trapped invader is valuable
        
        # 4. Bonus for blocking invader's escape route
        if self.enemy_border and invader_state.num_carrying > 0:
            invader_escape = min(self.enemy_border, 
                key=lambda b: self.get_maze_distance(invader_pos, b))
            my_dist_to_escape = self.get_maze_distance(my_pos, invader_escape)
            invader_dist_to_escape = self.get_maze_distance(invader_pos, invader_escape)
            
            if my_dist_to_escape <= invader_dist_to_escape:
                score += 100  # We're blocking their escape!
        
        # 5. Bonus for invader being in a choke point (easier to catch)
        if invader_pos in self.choke_points:
            score += 30
        
        # 6. Penalty for being a Pacman (crossed into enemy territory)
        if game_state.get_agent_state(self.index).is_pacman:
            score -= 200
        
        # 7. Bonus for being at a choke point that blocks escape
        if my_pos in self.choke_points:
            # Check if this choke is between invader and escape
            if self.enemy_border:
                closest_escape = min(self.enemy_border, 
                    key=lambda b: self.get_maze_distance(invader_pos, b))
                my_to_escape = self.get_maze_distance(my_pos, closest_escape)
                invader_to_escape = self.get_maze_distance(invader_pos, closest_escape)
                invader_to_me = self.get_maze_distance(invader_pos, my_pos)
                
                # Check if we're roughly on the path
                if abs((invader_to_me + my_to_escape) - invader_to_escape) <= 2:
                    score += 50
        
        return score

    def action_priority(self, game_state, action, target_pos):
        """
        Helper to prioritize actions that move toward the target.
        Used for move ordering in minimax to improve alpha-beta pruning.
        """
        successor = self.get_successor(game_state, action)
        new_pos = successor.get_agent_position(self.index)
        return -self.get_maze_distance(new_pos, target_pos)

    def get_features(self, game_state, action):
        """
        Hand–crafted features describing how good a single defender action is.

        This is used by the reflex-style evaluation (features * weights) as a fast
        alternative to running a full minimax search every turn.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Basic defense feature
        features['on_defense'] = 1
        if my_state.is_pacman: 
            features['on_defense'] = 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position()]
        features['num_invaders'] = len(invaders)
        
        if invaders:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
            
            # NEW: Track invader carrying food
            max_carrying = max(a.num_carrying for a in invaders)
            features['invader_carrying'] = max_carrying
            
            # NEW: Check if we're blocking escape route
            closest_invader = min(invaders, key=lambda a: self.get_maze_distance(my_pos, a.get_position()))
            invader_pos = closest_invader.get_position()
            
            if self.enemy_border and closest_invader.num_carrying > 0:
                invader_escape = min(self.enemy_border, 
                    key=lambda b: self.get_maze_distance(invader_pos, b))
                my_dist_to_escape = self.get_maze_distance(my_pos, invader_escape)
                invader_dist_to_escape = self.get_maze_distance(invader_pos, invader_escape)
                
                if my_dist_to_escape < invader_dist_to_escape:
                    features['blocking_escape'] = 1
                else:
                    features['blocking_escape'] = 0
        else:
            features['invader_distance'] = 0
            features['invader_carrying'] = 0
            features['blocking_escape'] = 0
        
        # NEW: Distance to patrol points (when no invaders)
        if not invaders and self.patrol_points:
            dist_to_patrol = min(self.get_maze_distance(my_pos, p) for p in self.patrol_points)
            features['patrol_distance'] = dist_to_patrol
        else:
            features['patrol_distance'] = 0
        
        # NEW: Capsule defense
        capsules = self.get_capsules_you_are_defending(successor)
        if capsules:
            dist_to_capsule = min(self.get_maze_distance(my_pos, c) for c in capsules)
            features['capsule_distance'] = dist_to_capsule
        else:
            features['capsule_distance'] = 0
        
        # NEW: Choke point positioning
        if my_pos in self.choke_points:
            features['at_choke_point'] = 1
        else:
            features['at_choke_point'] = 0
        
        # NEW: Scared timer consideration
        if my_state.scared_timer > 0:
            features['scared'] = 1
            # If scared, penalize getting too close to invaders
            if invaders:
                min_dist = min(self.get_maze_distance(my_pos, a.get_position()) for a in invaders)
                if min_dist <= 2:
                    features['danger_when_scared'] = 1
                else:
                    features['danger_when_scared'] = 0
        else:
            features['scared'] = 0
            features['danger_when_scared'] = 0

        # Action penalties
        if action == Directions.STOP: 
            features['stop'] = 1
        else:
            features['stop'] = 0
            
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: 
            features['reverse'] = 1
        else:
            features['reverse'] = 0

        return features

    def get_weights(self, game_state, action):
        """
        Dynamic weights for the defensive features, depending on the situation.

        Roughly:
          - When invaders are present, we heavily reward getting close and blocking escape.
          - When scared, we flip the preference and try to keep distance instead.
          - With no invaders, we care mostly about patrolling and sitting on good tiles.
        """
        my_state = game_state.get_agent_state(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position()]
        
        # Base weights
        weights = {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -15,
            'invader_carrying': 50,      # Prioritize invaders with food
            'blocking_escape': 200,       # Big bonus for blocking escape
            'patrol_distance': -2,        # Move toward patrol when idle
            'capsule_distance': -1,       # Slight preference to stay near capsules
            'at_choke_point': 20,         # Good defensive positioning
            'scared': -50,                # Avoid being scared state
            'danger_when_scared': -500,   # Really avoid invaders when scared
            'stop': -100,
            'reverse': -2,
        }
        
        # Adjust weights based on situation
        if my_state.scared_timer > 0:
            # When scared, don't chase invaders - evade them
            weights['invader_distance'] = 10  # Actually prefer distance
            weights['on_defense'] = 50        # Less important to stay on defense
            weights['patrol_distance'] = -5   # Prioritize patrolling safely
        
        if invaders:
            # Invaders present - prioritize interception
            max_carrying = max(a.num_carrying for a in invaders)
            if max_carrying > 2:
                # High-value target - prioritize catching
                weights['invader_distance'] = -25
                weights['blocking_escape'] = 400
            
            # If close to invader, don't reverse
            my_pos = game_state.get_agent_position(self.index)
            closest_dist = min(self.get_maze_distance(my_pos, a.get_position()) for a in invaders)
            if closest_dist <= 3:
                weights['reverse'] = -50  # Stronger penalty for reversing when close
                weights['stop'] = -200    # Definitely don't stop
        else:
            # No invaders - patrol mode
            weights['patrol_distance'] = -5
            weights['capsule_distance'] = -3
        
        return weights
