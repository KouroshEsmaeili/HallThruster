
class Grid1D:

    def __init__(self, z_edge):

        ncells = len(z_edge) - 1
        self.edges = list(z_edge)  # store the original edges if needed


        z_cell = []
        for i in range(ncells):
            z_mid = 0.5*(z_edge[i+1] + z_edge[i])
            z_cell.append(z_mid)


        z_cell = [z_edge[0]] + z_cell + [z_edge[-1]]

        dz_edge = []
        for i in range(len(z_cell)-1):
            dz_edge.append(z_cell[i+1] - z_cell[i])



        base_dz_cell = []
        for i in range(ncells):
            base_dz_cell.append(z_edge[i+1] - z_edge[i])
        extended_dz_cell = [base_dz_cell[0]] + base_dz_cell + [base_dz_cell[-1]]

        num_cells_final = ncells + 2

        self.num_cells = num_cells_final
        self.cell_centers = z_cell
        self.dz_edge = dz_edge
        self.dz_cell = extended_dz_cell

    def __repr__(self):
        return (f"<Grid1D: num_cells={self.num_cells}, "
                f"edges={self.edges}, "
                f"cell_centers={self.cell_centers}, "
                f"dz_edge={self.dz_edge}, "
                f"dz_cell={self.dz_cell}>")

