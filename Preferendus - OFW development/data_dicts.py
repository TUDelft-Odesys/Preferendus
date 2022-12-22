"""
File with all the data input for the optimization that is not in the Excel
"""

CONSTANTS = {
    'NC': 9,
    'NQ': 1,
    'NP': 9,
    'W_steel': 78.5,  # kN/m3
    'W_water': 10.25,  # kN/m3
    'W_concrete': 26.50,  # kN/m3
    'density_seawater': 1025,
}

STEEL = {
    'Yield Stress': 250,  # MPa
    'Tensile Strength': 400,  # MPa
    'Es': 200e3,  # MPa
    'Specific_weight': 78.60  # kN/m3
}
