import pickle
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

class KnotDetector:
    def __init__(self) -> None:
        self.crossings_stack = []
        self.crossings = []
        self.eps = 0
        self.knot = []
        self.start_idx = float('inf')
        
    def reset(self):
        self.crossings_stack = []
        self.crossings = []
        self.knot = []
        self.start_idx = float('inf')

    def run_crossing_correction(self): 
        '''
        Corrects all pairs of conflicting crossings sharing the same location, deferring to higher-confidence value.
        '''
        # correct all crossings in self.crossings (prior to adding to stack)       
        for curr_idx in range(len(self.crossings)): 
            curr_crossing = self.crossings[curr_idx]
            curr_x, curr_y = curr_crossing['loc']
            curr_id = curr_crossing['ID']
            curr_confidence = curr_crossing['confidence']
            for prev_idx in range(curr_idx):
                prev_crossing = self.crossings[prev_idx]
                prev_x, prev_y = prev_crossing['loc']
                prev_id = prev_crossing['ID']
                if np.linalg.norm(np.array([curr_x, curr_y]) - np.array([prev_x, prev_y])) <= self.eps and prev_id == curr_id:
                    prev_confidence = prev_crossing['confidence']
                    if curr_confidence >= prev_confidence:
                        prev_crossing['ID'] = 1 - prev_id
                        prev_crossing['confidence'] = curr_crossing['confidence']
                        print("Crossing " + str(prev_crossing['crossing_idx']) + " corrected.")
                        print("Previous confidence: " + str(prev_confidence))
                        print("Current confidence: " + str(curr_confidence))
                        print()
                    else:
                        curr_crossing['ID'] = 1 - curr_id
                        curr_crossing['confidence'] = prev_crossing['confidence']
                        print("Crossing " + str(curr_crossing['crossing_idx']) + " corrected.")
                        print("Previous confidence: " + str(prev_confidence))
                        print("Current confidence: " + str(curr_confidence))
                        print()
                    break

    def run_crossing_cancellation(self, max_steps=0, to_convergence=True):
        '''
        Runs crossing cancellation to convergence or upto max_steps, whichever comes first.
        '''
        if to_convergence:
            max_steps = float('inf')
        steps = 0
        converged = False

        while steps < max_steps:
            # Reset crossing stack
            prev_crossings_stack = self.crossings_stack[:]
            self.crossings_stack = []
            for crossing in prev_crossings_stack:
                if not self.crossings_stack:
                    curr_x, curr_y = crossing['loc']
                    self.crossings_stack.append(crossing)
                    continue

                # R1: simplify if same crossing is immediately re-encountered (over -> under or under -> over)
                prev_crossing = self.crossings_stack.pop()
                prev_x, prev_y = prev_crossing['loc']
                curr_x, curr_y = crossing['loc']
                prev_id, curr_id = prev_crossing['ID'], crossing['ID']

                if np.linalg.norm(np.array([curr_x, curr_y]) - np.array([prev_x, prev_y])) <= self.eps:
                    continue

                # R2: simplify if UU/OO is encountered later as OO/UU (at the same locations, in no order) 
                if curr_id == prev_id:
                    curr_pos = self.get_crossing_pos(crossing, len(self.crossings_stack))
                    prev_pos = self.get_crossing_pos(prev_crossing, len(self.crossings_stack))

                    if curr_pos != -1 and prev_pos != -1 and abs(curr_pos - prev_pos) == 1:
                        min_pos = min(curr_pos, prev_pos)
                        self.crossings_stack = self.crossings_stack[:min_pos] + self.crossings_stack[min_pos + 2:]
                        continue

                self.crossings_stack.append(prev_crossing)
                self.crossings_stack.append(crossing)

            # Break early if convergence is reached
            converged = self.crossings_stack == prev_crossings_stack
            if converged:
                return
            steps += 1

    def encounter_crossing(self, crossing):
        '''
        Called every time a new crossing is encountered.
        Adds crossing to list of crossings.
        '''
        # crossing must be in following format: {'loc': (x, y), 'ID': 0/1, 'confidence': [0, 1], 'crossing_idx': [0, ..., n], 'pixels_idx': [0, ..., p]}
        # ID options: 0 (under), 1 (over)
        # skip if not a crossing
        if crossing['ID'] == 2:
            return
        self.crossings.append(crossing)

    def find_knot(self):
        self.run_crossing_correction()
        self.crossings_stack = self.crossings[:]
        self.run_crossing_cancellation()
        self.knot = self.find_knot_helper()
        return self.knot
    
    def find_knot_helper(self):
        if len(self.crossings_stack) < 3:
            return []
        
        for cross_idx in range(len(self.crossings_stack)):
            crossing = self.crossings_stack[cross_idx]
            next_cross_idx = self.find_later_crossing_idx(crossing, self.crossings_stack, cross_idx + 1)
    
            # check if no occurrence of this crossing is found later
            if next_cross_idx == -1:
                continue
            
            # check if each crossing in between is paired off
            if self.check_if_all_pairs(self.crossings_stack[cross_idx + 1:next_cross_idx]):
                continue

            # check if all intermediate crossings are undercrossings
            if all([intermediate_crossing['ID'] == 0 for intermediate_crossing in self.crossings_stack[cross_idx + 1:next_cross_idx]]):
                continue

            # check if first crossing in knot is an overcrossing 
            if self.crossings_stack[cross_idx]['ID'] == 1:
                continue
            
            start_idx, end_idx = self.crossings_stack[cross_idx]['crossing_idx'], self.crossings_stack[next_cross_idx]['crossing_idx']
            return self.crossings[start_idx:end_idx + 1]
        
        return []

    def check_if_all_pairs(self, crossings_in_consideration):
        '''
        Returns if all the crossings in consideration are paired up.
        '''
        # if odd number of crossings, can't be paired off
        if len(crossings_in_consideration) % 2 == 1:
            return False
        num_matched_pairs = 0
        for cross_idx in range(len(crossings_in_consideration)):
            crossing = crossings_in_consideration[cross_idx]
            next_cross_idx = self.find_later_crossing_idx(crossing, crossings_in_consideration, cross_idx + 1)
            # if a crossing match is found
            if next_cross_idx != -1:
                num_matched_pairs += 1
        return num_matched_pairs == len(crossings_in_consideration) // 2

    def find_later_crossing_idx(self, crossing, crossings_in_consideration, start_idx):
        '''
        Returns the index of the next sighting of crossing on the stack (from start_idx onwards)
        If crossing has not been seen later, returns -1.
        '''
        curr_x, curr_y = crossing['loc']
        curr_id = crossing['ID']
        curr_crossing_idx = crossing['crossing_idx']
        for pos in range(start_idx, len(crossings_in_consideration)):
            next_crossing = crossings_in_consideration[pos]
            next_x, next_y = next_crossing['loc']
            next_id = next_crossing['ID']
            next_crossing_idx = next_crossing['crossing_idx']
            if np.linalg.norm(np.array([curr_x, curr_y]) - np.array([next_x, next_y])) <= self.eps:
                return pos
        return -1      

    # def add_crossing_to_stack(self, crossing):
    #     '''
    #     Runs cancellation and optionally adds crossing to stack.
    #     '''
    #     if not self.crossings_stack:
    #         curr_x, curr_y = crossing['loc']
    #         self.crossings_stack.append(crossing)
    #         return
        
    #     # R1: simplify if same crossing is immediately re-encountered (over -> under or under -> over)
    #     # TODO: check popping
    #     prev_crossing = self.crossings_stack.pop()
    #     prev_x, prev_y = prev_crossing['loc']
    #     curr_x, curr_y = crossing['loc']
    #     prev_id, curr_id = prev_crossing['ID'], crossing['ID']

    #     if np.linalg.norm(np.array([curr_x, curr_y]) - np.array([prev_x, prev_y])) <= self.eps:
    #         return

    #     # R2: simplify if UU/OO is encountered later as OO/UU (at the same locations, in no order) 
    #     if curr_id == prev_id:
    #         curr_pos = self.get_crossing_pos(crossing, len(self.crossings_stack))
    #         prev_pos = self.get_crossing_pos(prev_crossing, len(self.crossings_stack))

    #         if curr_pos != -1 and prev_pos != -1 and abs(curr_pos - prev_pos) == 1:
    #             min_pos = min(curr_pos, prev_pos)
    #             self.crossings_stack = self.crossings_stack[:min_pos] + self.crossings_stack[min_pos + 1:]
    #             return

    #     self.crossings_stack.append(prev_crossing)
    #     self.crossings_stack.append(crossing)
    #     self.check_for_knot()

    # def check_for_knot(self) -> bool:
    #     '''
    #     Checks if the latest crossing results in a knot.
    #     Only accounts for trivial loops.
    #     '''
    #     # no knot encountered if < 3 crossings have been seen (?)
    #     # if len(self.crossings_stack) < 3:
    #     #     return False
            
    #     crossing = self.crossings_stack[-1]
    #     pos = self.get_crossing_pos(crossing, len(self.crossings_stack) - 1)
    #     # no knot encountered if crossing hasn't been seen before
    #     if pos == -1:
    #         return

    #     # no knot encountered if O...U (?)
    #     if self.crossings_stack[pos]['ID'] == 1 and self.crossings_stack[-1]['ID'] == 0:
    #         return

    #     # intermediate crossing = crossing in between start and end crossing (exclusive)
    #     # no knot encountered if every intermediate crossing is an undercrossing
    #     if all([intermediate_crossing['ID'] == 0 for intermediate_crossing in self.crossings_stack[pos + 1:-1]]):
    #         return
                    
    #     start_idx = self.crossings_stack[pos]['crossing_idx']
    #     # if no knots previously identified / knot in consideration is present earlier than previously identified knot,
    #     # update self.knot
    #     if not self.knot or start_idx < self.start_idx:
    #         end_idx = self.crossings_stack[-1]['crossing_idx']
    #         self.knot = self.crossings[start_idx:end_idx + 1]
    #         self.start_idx = start_idx
    #         self.crossings_stack = self.crossings_stack[:pos]


    # def get_crossing_pos(self, crossing, end_stack_idx) -> int:
    #     '''
    #     Returns the index of previous sighting on the stack (before end_stack_idx).
    #     If crossing has not been seen previously, returns -1.
    #     '''
    #     curr_x, curr_y = crossing['loc']
    #     curr_id = crossing['ID']
    #     curr_crossing_idx = crossing['crossing_idx']

    #     # print(crossing)
    #     # print(self.crossings_stack)
    #     # print()

    #     # only look at crossings prior to most recently added crossing
    #     for pos in range(end_stack_idx):
    #         prev_crossing = self.crossings_in_consideration[pos]
    #         prev_x, prev_y = prev_crossing['loc']
    #         prev_id = prev_crossing['ID']
    #         prev_crossing_idx = prev_crossing['crossing_idx']
    #         if np.linalg.norm(np.array([curr_x, curr_y]) - np.array([prev_x, prev_y])) <= self.eps:
    #             return pos
        
    #     return -1        

    def get_crossing_pos(self, crossing, end_stack_idx) -> int:
        '''
        Returns the index of previous sighting on the stack (before end_stack_idx).
        If crossing has not been seen previously, returns -1.
        '''
        curr_x, curr_y = crossing['loc']
        curr_id = crossing['ID']
        curr_crossing_idx = crossing['crossing_idx']

        # print(crossing)
        # print(self.crossings_stack)
        # print()

        # only look at crossings prior to most recently added crossing
        for pos in range(end_stack_idx):
            prev_crossing = self.crossings_stack[pos]
            prev_x, prev_y = prev_crossing['loc']
            prev_id = prev_crossing['ID']
            prev_crossing_idx = prev_crossing['crossing_idx']
            if np.linalg.norm(np.array([curr_x, curr_y]) - np.array([prev_x, prev_y])) <= self.eps:
                return pos
        
        return -1        

    # def find_knot_from_corrected_crossings(self):
    #     '''
    #     Returns knot, if it exists.
    #     '''
    #     for crossing in self.crossings:
    #         self.add_crossing_to_stack(crossing)
    #     return self.knot

    def determine_under_over(self, img):
        mask = np.ones(img.shape[:2])
        mask[img[:, :, 0] <= 100] = 0
        # kernel = np.ones((2, 2), np.uint8)/4
        # smooth = cv2.filter2D(img,-1,kernel)
        # erode_mask = cv2.erode(mask, kernel)
        # plt.imshow(smooth)
        # plt.savefig('smooth.png')

def test_knot_detector_basic_r2(detector):
    detector.crossings = [
        {'loc': (0, 0), 'ID': 0}, # U at P0
        {'loc': (10, 10), 'ID': 0}, # U at P1
        {'loc': (0, 0), 'ID': 1}, # O at P0
        {'loc': (10, 10), 'ID': 1}, # O at P1
    ]

    for i in range(len(detector.crossings)):
        detector.crossings[i]['crossing_idx'] = i
        detector.crossings[i]['pixels_idx'] = i
        detector.crossings[i]['confidence'] = 1

    assert detector.find_knot() == []

def test_knot_detector_basic_r1r2(detector):
    detector.crossings = [
        {'loc': (0, 0), 'ID': 0}, # U at P0
        {'loc': (10, 10), 'ID': 0}, # U at P1
        {'loc': (20, 20), 'ID': 0}, # U at P2
        {'loc': (20, 20), 'ID': 1}, # O at P2
        {'loc': (0, 0), 'ID': 1}, # O at P0
        {'loc': (10, 10), 'ID': 1}, # O at P1
    ]

    for i in range(len(detector.crossings)):
        detector.crossings[i]['crossing_idx'] = i
        detector.crossings[i]['pixels_idx'] = i
        detector.crossings[i]['confidence'] = 1

    assert detector.find_knot() == []

def test_knot_detector_complex_r1r2(detector):
    detector.crossings = [
        {'loc': (0, 0), 'ID': 0}, # U at A
        {'loc': (10, 10), 'ID': 0}, # U at B
        {'loc': (30, 30), 'ID': 1}, # O at C
        {'loc': (40, 40), 'ID': 0}, # U at D
        {'loc': (0, 0), 'ID': 1}, # O at A
        {'loc': (10, 10), 'ID': 1}, # O at B
        {'loc': (30, 30), 'ID': 0}, # U at C
        {'loc': (40, 40), 'ID': 1}, # O at D
    ]

    for i in range(len(detector.crossings)):
        detector.crossings[i]['crossing_idx'] = i
        detector.crossings[i]['pixels_idx'] = i
        detector.crossings[i]['confidence'] = 1

    assert detector.find_knot() == []

def test_knot_detector_double_overhand(detector):
    # DEF ABC ABC DEF
    detector.crossings = [
        {'loc': (0, 0), 'ID': 0}, # U at D
        {'loc': (10, 10), 'ID': 1}, # O at E
        {'loc': (20, 20), 'ID': 0}, # U at F

        {'loc': (30, 30), 'ID': 0}, # U at A
        {'loc': (40, 40), 'ID': 1}, # O at B
        {'loc': (50, 50), 'ID': 0}, # U at C

        {'loc': (30, 30), 'ID': 1}, # O at A
        {'loc': (40, 40), 'ID': 0}, # U at B
        {'loc': (50, 50), 'ID': 1}, # O at C

        {'loc': (0, 0), 'ID': 1}, # O at D
        {'loc': (10, 10), 'ID': 0}, # U at E
        {'loc': (20, 20), 'ID': 1}, # O at F
    ]

    for i in range(len(detector.crossings)):
        detector.crossings[i]['crossing_idx'] = i
        detector.crossings[i]['pixels_idx'] = i
        detector.crossings[i]['confidence'] = 1

    assert detector.find_knot() == detector.crossings[:10]

def test_knot_detector_overhand_loops(detector):
    # DEF ABC ABC DEF
    detector.crossings = [
        {'loc': (0, 0), 'ID': 0}, # U at D

        {'loc': (30, 30), 'ID': 0}, # U at A
        {'loc': (40, 40), 'ID': 1}, # O at B
        {'loc': (50, 50), 'ID': 0}, # U at C
        {'loc': (30, 30), 'ID': 1}, # O at A
        {'loc': (40, 40), 'ID': 0}, # U at B
        {'loc': (50, 50), 'ID': 1}, # O at C

        {'loc': (10, 10), 'ID': 0}, # U at E
        {'loc': (10, 10), 'ID': 1}, # O at E

        {'loc': (20, 20), 'ID': 1}, # O at F
        {'loc': (20, 20), 'ID': 0}, # U at F

        {'loc': (0, 0), 'ID': 1}, # O at D
    ]

    for i in range(len(detector.crossings)):
        detector.crossings[i]['crossing_idx'] = i
        detector.crossings[i]['pixels_idx'] = i
        detector.crossings[i]['confidence'] = 1

    assert detector.find_knot() == detector.crossings[1:5]

def test_knot_detector_overhand_loops_o_first(detector):
    # DEF ABC ABC DEF
    detector.crossings = [
        {'loc': (0, 0), 'ID': 0}, # U at D

        {'loc': (30, 30), 'ID': 1}, # O at A
        {'loc': (40, 40), 'ID': 0}, # U at B
        {'loc': (50, 50), 'ID': 0}, # U at C
        {'loc': (30, 30), 'ID': 0}, # U at A
        {'loc': (40, 40), 'ID': 1}, # O at B
        {'loc': (50, 50), 'ID': 1}, # O at C

        {'loc': (10, 10), 'ID': 0}, # U at E
        {'loc': (10, 10), 'ID': 1}, # O at E

        {'loc': (20, 20), 'ID': 1}, # O at F
        {'loc': (20, 20), 'ID': 0}, # U at F

        {'loc': (0, 0), 'ID': 1}, # O at D
    ]

    for i in range(len(detector.crossings)):
        detector.crossings[i]['crossing_idx'] = i
        detector.crossings[i]['pixels_idx'] = i
        detector.crossings[i]['confidence'] = 1

    assert detector.find_knot() == []

if __name__ == '__main__':
    detector = KnotDetector()  

    test_knot_detector_basic_r2(detector)
    test_knot_detector_basic_r1r2(detector)
    test_knot_detector_complex_r1r2(detector)
    test_knot_detector_overhand_loops(detector)
    test_knot_detector_overhand_loops_o_first(detector)
 
    
  