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
    Contains helper methods for map analysis, successor generation, and pathfinding.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.choke_points = []

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.choke_points = self._identify_choke_points(game_state)
        CaptureAgent.register_initial_state(self, game_state)

    def _identify_choke_points(self, game_state):
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
        if self.is_camped_entering(game_state, my_state, my_pos, active_ghosts, loop_level):
            candidate_actions = actions
            return self.monte_carlo_escape(game_state, candidate_actions,
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

    def is_camped_entering(self, game_state, my_state, my_pos, active_ghosts, loop_level):
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

    def monte_carlo_escape(self, game_state, candidate_actions,
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
            features['distance_to_home'] = self.distance_to_home(my_pos_succ)
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
    """
    
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.previous_food = []
        self.target = None
    
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.previous_food = self.get_food_you_are_defending(game_state).as_list()
    
    def choose_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        
        # Clear target if we reached it
        if my_pos == self.target:
            self.target = None
        
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position()]
        
        # 1. Intercept Visible Invader using Minimax
        if invaders:
            closest_invader = min(invaders, key=lambda a: self.get_maze_distance(my_pos, a.get_position()))
            
            # Get the invader's agent index
            invader_index = None
            for i in self.get_opponents(game_state):
                if game_state.get_agent_state(i).get_position() == closest_invader.get_position():
                    invader_index = i
                    break
            
            if invader_index is not None:
                # Use Minimax to predict best interception move
                _, best_action = self.minimax(game_state, invader_index, depth=3, 
                                            alpha=-float('inf'), beta=float('inf'), 
                                            is_maximizing=True)
                if best_action:
                    return best_action
        
        # 2. Food Memory: Check if food was eaten (invader now invisible)
        current_food = self.get_food_you_are_defending(game_state).as_list()
        
        if len(current_food) < len(self.previous_food):
            # Food was eaten! Find which one and target it
            eaten_food = set(self.previous_food) - set(current_food)
            self.target = eaten_food.pop()  # Target the eaten food location
        
        # Update food memory for next turn
        self.previous_food = current_food
        
        # 3. Determine patrol target
        if self.target is None:
            # No specific target, patrol border near food
            if current_food:
                # Find border position closest to our food
                width = game_state.data.layout.width
                if self.red:
                    border_x = width // 2 - 1
                else:
                    border_x = width // 2
                
                # Generate candidate border positions
                height = game_state.data.layout.height
                border_positions = [(border_x, y) for y in range(height) 
                                   if not game_state.has_wall(border_x, y)]
                
                if border_positions:
                    # Pick border position closest to food cluster center
                    avg_food_y = sum(y for x, y in current_food) / len(current_food)
                    self.target = min(border_positions, key=lambda pos: abs(pos[1] - avg_food_y))
                else:
                    self.target = self.start
            else:
                # No food left, return to start
                self.target = self.start
        
        # Move toward target
        action = self._get_safe_path(game_state, self.target)
        if action:
            return action
            
        return random.choice(game_state.get_legal_actions(self.index))

    def minimax(self, game_state, invader_index, depth, alpha, beta, is_maximizing):
        """
        Minimax with Alpha-Beta Pruning to predict invader movement and intercept.
        
        Returns: (best_value, best_action)
        """
        # Base case: depth limit reached
        if depth == 0:
            return self.evaluate_defensive_state(game_state, invader_index), None
        
        # Check if invader was caught or escaped
        invader_pos = game_state.get_agent_position(invader_index)
        if not invader_pos:  # Invader returned home (escaped)
            return -1000, None
        
        my_pos = game_state.get_agent_position(self.index)
        if my_pos == invader_pos:  # Caught!
            return 1000, None
        
        if is_maximizing:
            # Our turn (defender): maximize value (minimize distance to invader)
            max_value = -float('inf')
            best_action = None
            
            actions = game_state.get_legal_actions(self.index)
            # Move ordering: prioritize moving toward invader
            actions = sorted(actions, key=lambda a: -self.action_priority(game_state, a, invader_pos))
            
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
            # Invader's turn: minimize value (maximize distance from us)
            min_value = float('inf')
            
            invader_actions = game_state.get_legal_actions(invader_index)
            
            for action in invader_actions:
                successor = game_state.generate_successor(invader_index, action)
                value, _ = self.minimax(successor, invader_index, depth - 1, alpha, beta, True)
                
                min_value = min(min_value, value)
                beta = min(beta, value)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            return min_value, None

    def evaluate_defensive_state(self, game_state, invader_index):
        """
        Evaluation function for defensive Minimax.
        Returns a score where higher is better for the defender.
        """
        my_pos = game_state.get_agent_position(self.index)
        invader_pos = game_state.get_agent_position(invader_index)
        
        if not invader_pos:  # Invader escaped
            return -1000
        
        distance = self.get_maze_distance(my_pos, invader_pos)
        
        # Negative distance because closer is better
        # Add bonus if invader is far from their home
        invader_state = game_state.get_agent_state(invader_index)
        bonus = 0
        if invader_state.num_carrying > 0:
            bonus = invader_state.num_carrying * 10  # More valuable to catch if carrying food
        
        return -distance + bonus

    def action_priority(self, game_state, action, target_pos):
        """
        Helper to prioritize actions that move toward the target.
        Used for move ordering in minimax to improve alpha-beta pruning.
        """
        successor = self.get_successor(game_state, action)
        new_pos = successor.get_agent_position(self.index)
        return -self.get_maze_distance(new_pos, target_pos)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position()]
        features['num_invaders'] = len(invaders)
        
        if invaders:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'stop': -100,
            'reverse': -2,
        }
