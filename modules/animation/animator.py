from .atomic_lattice import AtomicLattice
from .trap import Trap, AtomState
from .move import Move, MoveType

import numpy as np
from experiment.utils.configuration import Configuration
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List
from io import BytesIO
import imageio
from tqdm import tqdm
from matplotlib.patches import Rectangle, Circle

class Animator:

    lattice: AtomicLattice

    def __init__(self, lattice: AtomicLattice, config = "default", config_override: dict = {}):
        self.lattice = lattice
        self._setup_configs(config, config_override)

    def animate(self, moves: List[Move], address: str):
        """Saves the atom movements to a gif file

        Args:
            moves (List[Move]): List of moves to be carried out on the trapped atoms
            address (str): Address of the gif file to be saved, including the file name and extension
            fps (float): Frames per second
            frames_per_move (int): Number of frames per movement
        """
        self._save_frames(moves, address)
        
    def _save_frames(self, moves: List[Move], address: str):
        file_name = 0 
        for move_iter in tqdm(range(len(moves))):
            move = moves[move_iter]
            if move_iter >= 0:
                frames = self._get_move_frames_and_apply_move(move, address, file_name)
                for i in range(len(frames)):
                    file_name += 1
            else:
                self.lattice.apply_move(move)



    def _get_frames(self, moves: List[Move]):
        frames = []
        for move_iter in tqdm(range(len(moves))):
            move = moves[move_iter]
            frames += self._get_move_frames_and_apply_move(move)

        return frames
    
    def _get_move_frames_and_apply_move(self, move: Move, address, file_name):
        frames = []
        title = "" 
        fixed_atoms, moving_atoms_src, moving_atoms_dst = self._apply_move_to_lattice(move)

        for i in range(self.frames_per_move + 1):
            fig, ax = plt.subplots(dpi=1500)

            ax.axis("off")

            plt.title(title)
            plt.xlim([self.x0 - self.padding, self.x0 + self.padding + self.x_scale * (self.lattice.N_x-1)])
            plt.ylim([self.y0 - self.padding, self.y0 + self.padding + self.y_scale * (self.lattice.N_y-1)])
            figure_width_cm = 4.3
            figure_height_cm = 2.8
            figure_width_in = figure_width_cm / 2.54
            figure_height_in = figure_height_cm / 2.54
            fig.set_size_inches((figure_height_in, figure_width_in))
    
            ax.set_aspect('equal')
            extract = (move.move_type == MoveType.EXTRACT or move.move_type == MoveType.IMPLANT) and i != self.frames_per_move
            self._plot_traps(fig, ax, extract)
            self._plot_atoms(fig, ax, move, fixed_atoms, moving_atoms_src, moving_atoms_dst, i)

            ax.invert_yaxis()

            plt.rc('pdf', fonttype=42)
            plt.savefig(f"{address}/{file_name + i:0{4}d}.pdf", format = "pdf")
            frames += [imageio.imread(f"{address}/{file_name + i:0{4}d}.pdf")]
            plt.close()

        return frames
    
    def _apply_move_to_lattice(self, move: Move):
        fixed_atoms = []
        for i in range(self.lattice.N_y):
            for j in range(self.lattice.N_x):
                if self.lattice.get_trap(j, i).is_occupied():
                    fixed_atoms.append((j, i))
        fixed_atoms = set(fixed_atoms)

        moved_atoms_src = self.lattice.apply_move(move)
        for src in moved_atoms_src:
            try:   
                fixed_atoms.remove(src)
            except:
                pass
        fixed_atoms = list(fixed_atoms)
        
        moved_atoms_dst = list(moved_atoms_src)
        if move.move_type == MoveType.UP:
            corr_x, corr_y = 0, -1
        elif move.move_type == MoveType.DOWN:
            corr_x, corr_y = 0, 1
        elif move.move_type == MoveType.LEFT:
            corr_x, corr_y = -1, 0
        elif move.move_type == MoveType.RIGHT:
            corr_x, corr_y = 1, 0

        if move.move_type not in [MoveType.EXTRACT, MoveType.IMPLANT]:
            for i in range(len(moved_atoms_dst)):
                moved_atoms_dst[i] = (moved_atoms_dst[i][0] + corr_x, moved_atoms_dst[i][1] + corr_y)
        
        return fixed_atoms, moved_atoms_src, moved_atoms_dst
    
    def _get_move_title(self, move: Move):
        if move.move_type == MoveType.EXTRACT:
            move_name = "Extract"
        elif move.move_type == MoveType.IMPLANT:
            move_name = "Implant"
        elif move.move_type == MoveType.RIGHT:
            move_name = "Right"
        elif move.move_type == MoveType.LEFT:
            move_name = "Left"
        elif move.move_type == MoveType.UP:
            move_name = "Up"
        elif move.move_type == MoveType.DOWN:
            move_name = "down"
        else:
            raise NotImplementedError("Unhandled move type")
        
        return f"{move_name} x={move.index_x} y={move.index_y} block={move.block_size} ee={move.extraction_extent}"
    
    def _plot_traps(self, fig, ax, extract: bool = False):
        quarter = self.lattice.N_y / 4
        for i in range(self.lattice.N_y):
            for j in range(self.lattice.N_x):
                circle = self._get_trap_circle(j, i, i >= quarter and i < 3 * quarter)
                ax.add_patch(circle)
                if self.lattice.get_trap(j, i).trap_state == AtomState.DYNAMIC_TRAP and not extract:
                    square = self._get_dynamic_square(j,i)
                    ax.add_patch(square)

    def _plot_atoms(self,
                    fig, 
                    ax, 
                    move: Move, 
                    fixed_atoms: List[tuple], 
                    moving_atoms_src: List[tuple], 
                    moving_atoms_dst: List[tuple], 
                    frame_index: int):
        for atom in fixed_atoms:
            x, y = atom[0], atom[1]
            atom_patch = self._get_static_circle(x, y) \
                if self.lattice.get_trap(x, y).atom_state == AtomState.STATIC_TRAP else \
                self._get_dynamic_circle(x, y)
            ax.add_patch(atom_patch)
        
        if move.move_type in [MoveType.EXTRACT, MoveType.IMPLANT]:
            for atom in moving_atoms_src:
                x, y = atom[0], atom[1]
                atom_patch = \
                    self._get_static_to_dynamic_circle(x, y, 1 - frame_index / self.frames_per_move) \
                    if move.move_type == MoveType.IMPLANT else \
                    self._get_static_to_dynamic_circle(x, y, frame_index / self.frames_per_move)
                
                ax.add_patch(atom_patch)
        else:
            t = frame_index / self.frames_per_move
            for src, dst in zip(moving_atoms_src, moving_atoms_dst):
                x = eval(self.transition_function, None, {
                    "x1": src[0],
                    "x2": dst[0],
                    "t": t
                })
                y = eval(self.transition_function, None, {
                    "x1": src[1],
                    "x2": dst[1],
                    "t": t
                })
                atom_patch = self._get_dynamic_circle(x, y)

                ax.add_patch(atom_patch)

    def _get_trap_circle(self, x: int, y: int, target) -> Circle:
        return Circle(self._get_coordinates_from_index(x, y),
                                self.trap_radius,
                                edgecolor= self.trap_target_edgecolor if target else self.trap_edgecolor, 
                                facecolor= self.trap_target_facecolor if target else self.trap_facecolor, 
                                linewidth=self.trap_lw)

    def _get_dynamic_square(self, x: int, y: int) -> Circle:
        return Rectangle([self._get_coordinates_from_index(x, y)[t] - 2.5 for t in [0,1]],
                                5,
                                5,
                                edgecolor=self.atom_dynamic_facecolor, 
                                facecolor=self.trap_facecolor, 
                                linewidth=self.trap_lw)

    def _get_static_circle(self, x: int, y: int) -> Circle:
        return Circle(self._get_coordinates_from_index(x, y),
                                self.atom_static_radius,
                                edgecolor=self.atom_static_edgecolor, 
                                facecolor=self.atom_static_facecolor, 
                                linewidth=self.atom_static_lw)
    
    def _get_dynamic_circle(self, x: int, y: int) -> Circle:
            return Rectangle([self._get_coordinates_from_index(x, y)[t] - self.atom_static_radius for t in [0,1]],
                                self.atom_static_radius *2,
                                self.atom_static_radius *2,
                                edgecolor=self.atom_dynamic_facecolor, 
                                facecolor=self.atom_dynamic_facecolor, 
                                linewidth=self.trap_lw)
    
    def _get_static_to_dynamic_circle(self, x: int, y: int, t: float) -> Circle:
        static_fc = np.array([int(self.atom_static_facecolor[i:i+2], 16) / 255.0 for i in (1, 3, 5)])
        dynamic_fc = np.array([int(self.atom_dynamic_facecolor[i:i+2], 16) / 255.0 for i in (1, 3, 5)])
        fc = eval(self.transition_function, None, {
            "x1": static_fc,
            "x2": dynamic_fc,
            "t": t
        })

        static_ec = np.array([int(self.atom_static_edgecolor[i:i+2], 16) / 255.0 for i in (1, 3, 5)])
        dynamic_ec = np.array([int(self.atom_dynamic_edgecolor[i:i+2], 16) / 255.0 for i in (1, 3, 5)])
        ec = eval(self.transition_function, None, {
            "x1": static_ec,
            "x2": dynamic_ec,
            "t": t
        })

        radius = eval(self.transition_function, None, {
            "x1": self.atom_static_radius,
            "x2": self.atom_dynamic_radius,
            "t": t
        })

        lw = eval(self.transition_function, None, {
            "x1": self.atom_static_lw,
            "x2": self.atom_dynamic_lw,
            "t": t
        })
        
        return Circle(self._get_coordinates_from_index(x, y),
                        radius,
                        edgecolor=ec, 
                        facecolor=fc, 
                        linewidth=lw)
    
    def _get_coordinates_from_index(self, x, y):
        return self.x0 + x * self.x_scale, self.y0 + y * self.y_scale
    
    def _setup_configs(self, config: str, config_override: dict):
        self.configs = Configuration.generate_configs(config, file_attr=__file__)
        self.configs.update(config_override)
        self._add_values_to_object(self.configs)
        
    def _add_values_to_object(self, dic: dict, prefix: str = ""):
        for k, v in dic.items():
            if type(v) == dict:
                self._add_values_to_object(v, prefix = prefix + k + "_")
            else:
                setattr(self, prefix + k, v)
