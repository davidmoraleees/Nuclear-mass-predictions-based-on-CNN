from src.utils.config import load_config

def load_constants():
    """Load YAML configuration and return both raw config and a computed constants dictionary."""
    cfg = load_config()
    uma = cfg["LDM"]["uma"]
    const = {
        "uma": uma,
        "m_n": cfg["LDM"]["m_n"] * 1e-6 * uma,
        "m_H": cfg["LDM"]["m_H"] * 1e-6 * uma,
        "m_e": cfg["LDM"]["m_e"],
        "av": cfg["LDM"]["av"],
        "aS": cfg["LDM"]["aS"],
        "ac": cfg["LDM"]["ac"],
        "aA": cfg["LDM"]["aA"],
        "ap": cfg["LDM"]["ap"],
        "N_MIN": cfg["general"]["N_MIN"],
        "N_MAX": cfg["general"]["N_MAX"],
        "Z_MIN": cfg["general"]["Z_MIN"],
        "Z_MAX": cfg["general"]["Z_MAX"],
        "BE_C1": cfg["general"]["BE_C1"],
        "BE_C2": cfg["general"]["BE_C2"],
    }
    return cfg, const


def in_region(N, Z, const):
    """Return a boolean mask selecting nuclides inside the (N,Z) region bounds from config."""
    return (N >= const["N_MIN"]) & (N < const["N_MAX"]) & (Z >= const["Z_MIN"]) & (Z < const["Z_MAX"])


def b_e(Z, const):
    """Compute electron binding energy correction b_e(Z) using the configured coefficients."""
    return (const["BE_C1"] * (Z**2.39) + const["BE_C2"] * (Z**5.35)) * 1e-6
