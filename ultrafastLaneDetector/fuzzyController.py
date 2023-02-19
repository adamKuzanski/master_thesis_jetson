import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class LaneDepartureDetector:
    def __init__(self):
        distance_to_left = ctrl.Antecedent(np.arange(0, 300, 1), 'distance_left')
        distance_to_right = ctrl.Antecedent(np.arange(0, 300, 1), 'distance_right')
        lane_departure = ctrl.Consequent(np.arange(0, 100, 1), 'lane_departure')
        delta_left_lane = ctrl.Antecedent(np.arange(-40, 40, 1), 'delta_left_lane')

        lane_departure['not_significant'] = fuzz.gaussmf(lane_departure.universe, 10, 10)
        lane_departure['warning'] = fuzz.gaussmf(lane_departure.universe, 20, 15)
        lane_departure['danger'] = fuzz.trimf(lane_departure.universe, [30, 101, 101])

        distance_to_left.automf(number=3, names=["touching", "close", "far"])
        distance_to_right.automf(number=3, names=["touching", "close", "far"])
        delta_left_lane.automf(number=3, names=["stop", "slow", "fast"])  

        rule1 = ctrl.Rule(distance_to_left["touching"] & distance_to_right["far"] & (delta_left_lane["slow"] | delta_left_lane["fast"]), lane_departure["danger"])
        rule2 = ctrl.Rule(distance_to_left["far"] & distance_to_right["touching"] & (delta_left_lane["slow"] | delta_left_lane["fast"]), lane_departure["danger"])

        rule3 = ctrl.Rule(distance_to_left["touching"] & distance_to_right["close"]  & (delta_left_lane["slow"] | delta_left_lane["fast"]), lane_departure["danger"])
        rule4 = ctrl.Rule(distance_to_left["close"] & distance_to_right["touching"]  & (delta_left_lane["slow"] | delta_left_lane["fast"]), lane_departure["danger"])

        rule5 = ctrl.Rule(distance_to_left["close"] & distance_to_right["far"], lane_departure["warning"])
        rule6 = ctrl.Rule(distance_to_left["far"] & distance_to_right["close"], lane_departure["warning"])

        rule7 = ctrl.Rule(distance_to_left["close"] & distance_to_right["close"], lane_departure["not_significant"])
        rule8 = ctrl.Rule(distance_to_left["far"] & distance_to_right["far"], lane_departure["not_significant"])
        rule9 = ctrl.Rule(distance_to_left["touching"] & distance_to_right["touching"], lane_departure["not_significant"])
        
        self.lane_crossing_detector = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        self.crossing = ctrl.ControlSystemSimulation(self.lane_crossing_detector)

    def calculate(self, dist_left, dist_right, delta_left):
        self.crossing.input['distance_left'] = dist_left
        self.crossing.input['distance_right'] = dist_right
        self.crossing.input['delta_left_lane'] = delta_left
        self.crossing.compute()
        return self.crossing.output['lane_departure']