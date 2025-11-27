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
    A reflex agent that seeks food using A* pathfinding.
    
    Strategy:
    1. Safety First: If a ghost is nearby (within 5 steps), switch to reactive mode to survive.
    2. Goal Selection:
       - If carrying a lot of food or time is running out -> Return Home.
       - If ghosts are chasing us -> Go for a Power Capsule.
       - Otherwise -> Go for the nearest Food.
    3. Pathfinding: Use A* to find a safe path to the selected goal, avoiding ghosts.
    """
    def choose_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        
        # 1. Safety First (Expectimax-lite)
        # If a ghost is too close, fall back to feature-based reactive movement
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() and a.scared_timer < 5]
        if ghosts:
            min_dist = min([self.get_maze_distance(my_pos, g.get_position()) for g in ghosts])
            if min_dist <= 5:
                return self._choose_reactive_action(game_state)

        # 2. Strategic Goal Selection
        target = None
        
        # Return home logic
        carry_amount = game_state.get_agent_state(self.index).num_carrying
        time_left = game_state.data.timeleft
        if carry_amount > 0 and (carry_amount >= 5 or time_left < 200 or not food_list):
             target = self.start 
             
        # Capsule logic
        if not target and capsules and ghosts:
            target = min(capsules, key=lambda c: self.get_maze_distance(my_pos, c))
            
        # Food logic
        if not target and food_list:
            target = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
            
        # 3. Pathfinding
        action = self._get_safe_path(game_state, target)
        if action:
            return action
            
        return self._choose_reactive_action(game_state)

    def _choose_reactive_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        
        features['successor_score'] = -len(self.get_food(successor).as_list())

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() and a.scared_timer < 5]
        if ghosts:
            dists = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]
            min_dist = min(dists)
            if min_dist <= 1: features['ghost_distance'] = -1000
            elif min_dist <= 2: features['ghost_distance'] = -100
            
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'ghost_distance': 1}

##########
# Defensive Agent #
##########

class SmartDefensiveAgent(ReflexCaptureAgent):
    """
    A defensive agent that patrols and intercepts invaders.
    
    Strategy:
    1. Intercept: If an invader is seen, calculate a safe path to intercept them immediately.
    2. Patrol: If no enemies are visible, patrol key "choke points" (narrow passages) to block entry.
    """
    def choose_action(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position()]
        
        # 1. Intercept Invader
        if invaders:
            # Simple chase logic: move towards the closest invader
            closest_invader = min(invaders, key=lambda a: self.get_maze_distance(game_state.get_agent_position(self.index), a.get_position()))
            action = self._get_safe_path(game_state, closest_invader.get_position())
            if action:
                return action
            
        # 2. Patrol Choke Points
        if not self.choke_points:
            self.choke_points = self._identify_choke_points(game_state)
            
        target = self.start
        if self.choke_points:
            target = random.choice(self.choke_points)
            
        action = self._get_safe_path(game_state, target)
        if action:
            return action
            
        return random.choice(game_state.get_legal_actions(self.index))

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
