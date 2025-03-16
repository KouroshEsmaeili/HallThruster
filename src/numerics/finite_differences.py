def forward_diff_coeffs(x0, x1, x2):

    h1 = x1 - x0
    h2 = x2 - x1
    c0 = -(2.0*h1 + h2)/(h1*(h1 + h2))
    c1 = (h1 + h2)/(h1*h2)
    c2 = -h1/(h2*(h1 + h2))
    return (c0, c1, c2)

def central_diff_coeffs(x0, x1, x2):

    h1 = x1 - x0
    h2 = x2 - x1
    c0 = -(h2)/(h1*(h1 + h2))
    c1 = -((h1 - h2)/(h1*h2))
    c2 = h1/(h2*(h1 + h2))
    return (c0, c1, c2)

def backward_diff_coeffs(x0, x1, x2):

    h1 = x1 - x0
    h2 = x2 - x1
    c0 = h2/(h1*(h1 + h2))
    c1 = -((h1 + h2)/(h1*h2))
    c2 = (h1 + 2.0*h2)/(h2*(h1 + h2))
    return (c0, c1, c2)

def second_deriv_coeffs(x0, x1, x2):
    h1 = x1 - x0
    h2 = x2 - x1
    common = 2.0/(h1*h2*(h1 + h2))
    return (h2*common, -((h1 + h2)*common), h1*common)

def upwind_diff_coeffs(x0, x1, x2):
    h1 = x1 - x0
    return (-1.0/h1, 1.0/h1, 0.0)

def downwind_diff_coeffs(x0, x1, x2):
    h2 = x2 - x1
    return (0.0, -1.0/h2, 1.0/h2)



def forward_difference(f0, f1, f2, x0, x1, x2):
    c0, c1, c2 = forward_diff_coeffs(x0, x1, x2)
    return c0*f0 + c1*f1 + c2*f2

def central_difference(f0, f1, f2, x0, x1, x2):
    c0, c1, c2 = central_diff_coeffs(x0, x1, x2)
    return c0*f0 + c1*f1 + c2*f2

def backward_difference(f0, f1, f2, x0, x1, x2):
    c0, c1, c2 = backward_diff_coeffs(x0, x1, x2)
    return c0*f0 + c1*f1 + c2*f2

def second_deriv_central_diff(f0, f1, f2, x0, x1, x2):
    c0, c1, c2 = second_deriv_coeffs(x0, x1, x2)
    return c0*f0 + c1*f1 + c2*f2



def interpolation_coeffs(x, x0, x1):
    denom = (x1 - x0)
    if denom == 0.0:
        # avoid division by zero
        print('[interpolation_coeffs] denom isc getting zero', denom)
        return (0.0, 1.0)  # or some fallback
    c0 = (x - x0)/denom
    c1 = 1.0 - c0
    return (c0, c1)