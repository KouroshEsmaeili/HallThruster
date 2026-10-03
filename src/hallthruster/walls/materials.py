from dataclasses import dataclass

@dataclass
class WallMaterial:
    name: str
    e_star: float
    sigma0: float

def SEE_yield(material: WallMaterial, Tev: float, gamma_max: float) -> float:
    # Unpack material properties.
    sigma0 = material.sigma0
    e_star = material.e_star
    gamma = sigma0 + 1.5 * Tev / e_star * (1 - sigma0)
    return min(gamma_max, gamma)

# --- Predefined Wall Materials ---
Alumina = WallMaterial(name="Alumina", e_star=22.0, sigma0=0.57)
BoronNitride = WallMaterial(name="Boron Nitride", e_star=45.0, sigma0=0.24)
SiliconDioxide = WallMaterial(name="Silicon Dioxide", e_star=18.0, sigma0=0.5)
BNSiO2 = WallMaterial(name="Boron Nitride Silicon Dioxide", e_star=40.0, sigma0=0.54)
SiliconCarbide = WallMaterial(name="Silicon Carbide", e_star=43.0, sigma0=0.69)

# --- Dictionary of Wall Materials for Serialization/Reference ---
wall_materials = {
    "Alumina": Alumina,
    "BoronNitride": BoronNitride,
    "SiliconDioxide": SiliconDioxide,
    "BNSiO2": BNSiO2,
    "SiliconCarbide": SiliconCarbide,
}

def get_wall_materials():
    return wall_materials
