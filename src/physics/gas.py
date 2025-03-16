from .physicalconstants import R0, NA


class Gas:
    def __init__(self, name, short_name, gamma, M):
        self.name = name
        self.short_name = short_name
        self.gamma = gamma
        self.M = M

        self.R = R0 / M
        self.m = M / NA  # in kg
        self.cp = self.gamma/(self.gamma - 1)*self.R
        self.cv = self.cp - self.R

    def __repr__(self):
        return f"<Gas {self.name}>"

    def __call__(self, Z):

        return Species(self, Z)


class Species:

    def __init__(self, element: Gas, Z: int):
        self.element = element
        self.Z = Z
        self.symbol = self.species_string(element, Z)

    def species_string(self, element, Z):
        if Z == 0:
            return element.short_name
        sign_str = "+" if Z > 0 else "-"
        if abs(Z) > 1:
            return element.short_name + str(abs(Z)) + sign_str
        return element.short_name + sign_str

    def __repr__(self):
        return f"{self.symbol}"

    def __eq__(self, other):
        if not isinstance(other, Species):
            return NotImplemented
        return (self.element == other.element) and (self.Z == other.Z)

    def __lt__(self, other):
        if not isinstance(other, Species):
            return NotImplemented
        # If the element is the same, compare by Z; otherwise compare by element string
        if self.element == other.element:
            return self.Z < other.Z
        return self.element < other.element

    def __hash__(self):
        return hash((self.element, self.Z))

Argon = Gas("Argon", "Ar", gamma=5/3, M=39.948)
Krypton = Gas("Krypton", "Kr", gamma=5/3, M=83.798)
Xenon = Gas("Xenon", "Xe", gamma=5/3, M=131.293)
Bismuth = Gas("Bismuth", "Bi", gamma=5/3, M=208.9804)
Mercury = Gas("Mercury", "Hg", gamma=5/3, M=200.59)

propellants = {
    "Xenon": Xenon,
    "Krypton": Krypton,
    "Argon": Argon,
    "Bismuth": Bismuth,
    "Mercury": Mercury
}
