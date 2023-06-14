from typing import List, Tuple

class PacmanState:
    def __init__(self, pacman_pos: Tuple[int, int], ghosts_pos: List[Tuple[int, int]]):
        self.pacman_pos = pacman_pos
        self.ghosts_pos = sorted(ghosts_pos, key=lambda t: (t[1], t[0]))
        self.vector = [self.pacman_pos] + self.ghosts_pos
        # todo dodac dwa znaczniki czy są kulki mocy aktywne w stylu dodanie na koncu 0/1,0/1
    def get_state(self) -> Tuple:
        return tuple(item for sublist in self.vector  for item in sublist)