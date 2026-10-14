from typing import Dict, List
import numpy as np

def logit(data, bounds, inv_transform=False):
    if not inv_transform:
        transform = np.log((data-bounds[0])/(bounds[1]-data))
    elif inv_transform:
        transform = (bounds[1]*np.exp(data)+bounds[0])/(1+np.exp(data))
    return transform

def chirp_mass(m1, m2):
    """Calculate chirp mass from component masses"""
    return ((m1 * m2) ** 3 / (m1 + m2)) ** (1 / 5)

def cm_mr_to_m1_m2(cm, mr):
    """Calculate component masses from chirp mass and mass ratio"""
    return cm*((1+mr)**(1/5))*(mr**(-3/5)), cm*((1+mr)**(1/5))*(mr**(2/5))

def mr_limit_from_cm(cm, min_m2):
    """Calculate minimum mass ratio given chirp mass and minimum m2"""
    y = min_m2/cm
    t = y**5
    if y < (4/27)**(.2):
        return -1/3+(2/3)*np.cos((1/3)*np.arccos((27/2)*t-1))
    else:
        a = t/2-1/27
        b = ((t**2)/4-t/27)**(1/2)
        return (a+b)**(1/3)+(a-b)**(1/3)-1/3

def mass_ratio_condition_powerlaw(reference_params, cm):
    """
    Return a dictionary that can be interpreted by Bilby's
    `ConditionalUniform` to set the minimum value of `mr`
    from `cm`
    """
    return {
        "alpha": reference_params["alpha"],
        "minimum": mr_limit_from_cm(cm, 5),
        "maximum": reference_params["maximum"],
    }

#l = 0.5881736474620177, mean = 1.5213759844071217
def yj_transform(y, l = 0.5881736474620177, mean = 1.5213759844071217, scale = 1.021261470492894): #For single use transform
    if y >= 0:
        raw = (((y+1)**l)-1)/l
    else:
        raw = -((-y+1)**(2-l)-1)/(2-l)
    return (raw - mean) / scale
   
logit_bounds = [0, 1]
def mass_ratio_condition_truncatedgaussian(reference_params, cm):
    """
    Return a dictionary that can be interpreted by Bilby's
    `ConditionalUniform` to set the minimum value of `mr`
    from `cm`
    """
    #if np.isnan(pt.transform([[logit(mr_limit_from_cm(cm, 5).real, [0, 1])]]).squeeze(axis=1)).any():
    #    print(logit(mr_limit_from_cm(cm, 5).real, [0, 1]))
    #    print(mr_limit_from_cm(cm, 5).real)
    #    print(cm)
    return {
        "mu": reference_params["mu"],
        "sigma": reference_params["sigma"],
        "minimum": yj_transform(logit(mr_limit_from_cm(cm, 5).real, [0, 1])),
        "maximum": reference_params["maximum"],
    }

def mass_ratio_condition_gamma(reference_params, cm):
    """
    Return a dictionary that can be interpreted by Bilby's
    `ConditionalUniform` to set the minimum value of `mr`
    from `cm`
    """
    return {
        "k": reference_params["mu"],
        "theta": reference_params["sigma"],
    }

def mass_condition_uniform(reference_params, mass_1):
    """
    Return a dictionary that can be interpreted by Bilby's
    `ConditionalUniform` to set the maximum value of `mass_2`
    to be whatever was drawn for `mass_1`
    """
    return {
        "minimum": reference_params["minimum"],
        "maximum": mass_1,
    }


def mass_condition_powerlaw(reference_params, mass_1):
    """
    Return a dictionary that can be interpreted by Bilby's
    `ConditionalPowerLaw` to set the maximum value of `mass_2`
    to be whatever was drawn for `mass_1`
    """
    return {
        "alpha": reference_params["alpha"],
        "minimum": reference_params["minimum"],
        "maximum": mass_1,
    }


def mass_constraints(samples):
    """
    Return a dictionary with new keys `mass_ratio` and `chirp_mass`
    so that Bilby `Constraint`s on these keys will be effected
    """
    if "mass_1" not in samples or "mass_2" not in samples:
        raise KeyError("mass_1 and mass_1 must exist to have a mass_ratio")
    samples["mass_ratio"] = samples["mass_2"] / samples["mass_1"]
    samples["chirp_mass"] = chirp_mass(samples["mass_1"], samples["mass_2"])
    return samples


def transpose(d: Dict[str, List]):
    """Turn a dict of lists into a list of dicts"""
    return [
        dict(zip(d, col, strict=True)) for col in zip(*d.values(), strict=True)
    ]
