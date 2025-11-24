# baseline_team.py
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


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.choke_points = self.identify_choke_points(game_state)
        CaptureAgent.register_initial_state(self, game_state)

    def identify_choke_points(self, game_state):
        """
        Analyzes the map to identify narrow corridors (choke points).
        """
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
        """
        Finds the next state after applying the given action.
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights.
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        return util.Counter()

    def get_weights(self, game_state, action):
        return {}

##########
# Offensive Agent #
##########

class SmartOffensiveAgent(ReflexCaptureAgent):
    """
    Smart Offensive Agent:
    - Uses safety checks to avoid suicide.
    - Tracks visited states to prevent looping.
    - Dynamically adjusts weights based on carrying amount.
    """
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.mode = "attack" # Start in attack
        self.visited_positions = util.Counter() # Tracks visited positions to prevent loops

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        
        # 1. Update Visited History
        # Reset history if we eat food or return home to allow retracing productive paths
        if game_state.get_agent_state(self.index).num_returned > 0:
            self.visited_positions.clear()
        self.visited_positions[my_pos] += 1

        # 2. Mode Switching Logic
        if self.mode == "attack":
            # Logic to switch modes can be expanded here
            pass
            
        # 3. Get all legal actions
        actions = game_state.get_legal_actions(self.index)
        
        # 4. Safety Layer
        # Filter actions immediately to prevent suicide.
        safe_actions = self.get_safe_actions(game_state, actions)
        
        if not safe_actions:
            # If trapped, stop is the default fallback
            return Directions.STOP
            
        # 5. Decision Logic
        # Compute scores for all SAFE actions.
        best_action = self.compute_best_action(game_state, safe_actions)
        
        return best_action

    def get_safe_actions(self, game_state, actions):
        """
        Improved Safety Logic:
        1. Filters out actions that lead directly to death.
        2. If ALL actions are unsafe, picks the one that maximizes survival time (distance to ghost).
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer == 0]

        safe_actions = []
        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_position(self.index)
            
            is_safe = True
            for defender in defenders:
                # Keep a buffer of 2 steps
                if self.get_maze_distance(successor_pos, defender.get_position()) <= 2:
                    is_safe = False
                    break
            if is_safe:
                safe_actions.append(action)
                
        # If trapped, don't stop! Run to the furthest point from the ghost.
        if not safe_actions and len(defenders) > 0:
            # Find the closest defender (the immediate threat)
            closest_defender = min(defenders, key=lambda d: self.get_maze_distance(game_state.get_agent_position(self.index), d.get_position()))
            
            # Pick action that maximizes distance to that defender
            best_panic_action = max(actions, key=lambda a: self.get_maze_distance(
                self.get_successor(game_state, a).get_agent_position(self.index), 
                closest_defender.get_position()
            ))
            return [best_panic_action]

        return safe_actions if safe_actions else actions

    def compute_best_action(self, game_state, actions):
        """
        Evaluates Q(s,a) = w * f for all actions and picks the max.
        """
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        
        # Tie-breaking: prefer forward motion over stopping
        if Directions.STOP in best_actions and len(best_actions) > 1:
            best_actions.remove(Directions.STOP)
            
        return random.choice(best_actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_list = self.get_food(successor).as_list()

        # Feature 1: Successor Score
        features['successor_score'] = -len(food_list)

        # Feature 2: Distance to Food
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Feature 3: Return Home
        # If carrying food, returning home becomes valuable
        if my_state.num_carrying > 0:
            features['distance_to_home'] = self.get_maze_distance(my_pos, self.start)

        # Feature 4: Visited States (Loop breaker)
        # Penalize going where we have already been recently
        if my_pos in self.visited_positions:
            features['visited_penalty'] = self.visited_positions[my_pos]

        # Feature 5: Distance to Capsules
        capsules = self.get_capsules(successor)
        if len(capsules) > 0:
            features['distance_to_capsule'] = min([self.get_maze_distance(my_pos, c) for c in capsules])

        return features

    def get_weights(self, game_state, action):
        my_state = self.get_successor(game_state, action).get_agent_state(self.index)
        
        # Dynamic Weighting
        food_weight = -2
        home_weight = 0 # Default: don't care about home
        
        # If carrying a little, still focus on food
        if my_state.num_carrying < 3:
            food_weight = -4  # Hunger increases
            home_weight = 0   # Ignore home
            
        # If carrying a lot, start thinking about home
        elif my_state.num_carrying >= 3:
            food_weight = -1
            home_weight = -2
            
        # If carrying a TON or time is running out, RUN HOME
        if my_state.num_carrying > 5 or game_state.data.timeleft < 200:
            home_weight = -10 

        return {
            'successor_score': 100,
            'distance_to_food': food_weight,
            'distance_to_home': home_weight,
            'distance_to_capsule': -2,
            'visited_penalty': -10, 
        }

##########
# Defensive Agent #
##########

class SmartDefensiveAgent(ReflexCaptureAgent):
    """
    Smart Defensive Agent:
    - Prioritizes visible invaders.
    - Uses noisy distance readings to infer invisible invader positions.
    - Patrols defended food when no enemies are detected.
    """
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.center_position = None

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        mid_x = (width // 2) - 1 if self.red else (width // 2)
        self.center_position = (mid_x, height // 2)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # 1. On Defense?
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # 2. Visible Invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            # 3. Invisible Logic: Patrol the FOOD being defended
            # We check the noisy distance, but we also check our food
            noisy_distances = game_state.get_agent_distances()
            opponent_indices = self.get_opponents(successor)
            opp_dists = [noisy_distances[i] for i in opponent_indices]
            
            if len(opp_dists) > 0:
                 features['invisible_invader_distance'] = min(opp_dists)
            
            # Patrol the food clusters
            # Get list of food we are defending
            food_defending = self.get_food_you_are_defending(successor).as_list()
            if len(food_defending) > 0:
                 # Find the food closest to the 'center' or average position
                 # Simple version: minimize distance to nearest food dot
                 features['distance_to_defended_food'] = min([self.get_maze_distance(my_pos, f) for f in food_defending])

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'invisible_invader_distance': -2, 
            'distance_to_defended_food': -1, # Patrol Weight
            'stop': -100,
            'reverse': -2,
        }