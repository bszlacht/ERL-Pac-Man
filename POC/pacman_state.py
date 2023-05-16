from typing import List, Tuple

class PacmanState:
    def __init__(self, pacman_pos: Tuple[int, int], ghosts_pos: List[Tuple[int, int]]):
        self.pacman_pos = pacman_pos
        self.ghosts_pos = sorted(ghosts_pos, key=lambda t: (t[1], t[0]))
    def __hash__(self) -> int:
        return hash(self.pacman_pos) * 13 + hash(tuple(self.ghosts_pos))
    def __eq__(self, __value: object) -> bool:
      if __value is None:
         return False
      return __value.pacman_pos == self.pacman_pos and self.ghosts_pos == __value.ghosts_pos