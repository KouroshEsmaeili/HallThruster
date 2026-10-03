def units(x):
    return 1

def convert_to_float64(number, unit):
    return float(number)


# Example usage:
if __name__ == "__main__":
    print("units(123) =", units(123))
    print("convert_to_float64(3, 'm') =", convert_to_float64(3, 'm'))
