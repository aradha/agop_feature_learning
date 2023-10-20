from neural_tangents import stax as stax
import sys
import functools
from math import sqrt

def nonlinearity():
    return stax.ABRelu(b=sqrt(2), a=0)

def Vanilla(width=1, ps=3, num_classes=1, depth=1, expanded=False):
    mode = 'VALID'
    if expanded:
        mode = 'VALID'
        layers = [
                    stax.Conv(width, (ps,ps), strides=(ps,ps), W_std=ps, b_std=0, padding=mode),
                    nonlinearity()
                ]
    else:
        layers = [
                    stax.Conv(width, (ps,ps), strides=(1,1), W_std=ps, b_std=0, padding=mode),
                    nonlinearity()
                ]


    for _ in range(depth-1):
        layers += [
                    stax.Conv(width, (ps,ps), strides=(1,1), W_std=ps, b_std=0, padding=mode),
                    nonlinearity()
                  ]

    layers += [
            stax.Flatten(),
            stax.Dense(num_classes, W_std=1)
        ]
    
    return stax.serial(*layers) 
