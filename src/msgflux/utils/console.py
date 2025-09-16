def cprint(text: str, *, lc=None, bc=None, ls=None, **kwargs):
    """Colored print.
    
    Args:
        lc (str): Line color (foreground).
        bc (str): Background color.
        ls (str): Line style.
    """
    FG_COLORS = {
        "k": 30,  # black
        "r": 31,  # red
        "g": 32,  # green
        "y": 33,  # yellow
        "b": 34,  # blue
        "m": 35,  # magenta
        "c": 36,  # cyan
        "w": 37,  # white
    }

    BG_COLORS = {
        "k": 40,
        "r": 101,
        "g": 102,
        "y": 103,
        "b": 44,
        "m": 45,
        "c": 46,
        "w": 47,
        # extra light tones
        "br1": "48;5;180",  # light brown
        "br2": "48;5;222",  # beige
        "br3": "48;5;223",  # sand
        "br4": "48;5;229",  # cream
    }

    STYLES = {
        "b": 1,  # bold
        "i": 3,  # italic
        "u": 4,  # underline
        "v": 7,  # inverse
    }

    codes = []

    if lc in FG_COLORS:
        codes.append(str(FG_COLORS[lc]))
    if bc in BG_COLORS:
        codes.append(str(BG_COLORS[bc]))
    if ls in STYLES:
        codes.append(str(STYLES[ls]))

    prefix = f"\033[{';'.join(codes)}m" if codes else ""
    suffix = "\033[0m"

    print(f"{prefix}{text}{suffix}", **kwargs)
