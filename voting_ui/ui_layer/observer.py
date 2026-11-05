"""Независимый наблюдатель: покоординатная интерполяция опубликованных точек."""
from __future__ import annotations


def interpolate_coefficients(points: dict[int,int], p: int) -> tuple[int,...]:
    if not points or len({x%p for x in points})!=len(points):
        raise ValueError('Точки должны быть непустыми и различными в F_p')
    out=[0]*len(points)
    for x,y in points.items():
        basis=[1]
        denom=1
        for z in points:
            if z==x: continue
            b=[0]*(len(basis)+1)
            for d,value in enumerate(basis):
                b[d]=(b[d]-z*value)%p
                b[d+1]=(b[d+1]+value)%p
            basis=b
            denom=(denom*(x-z))%p
        factor=(y*pow(denom,-1,p))%p
        for d,value in enumerate(basis):
            out[d]=(out[d]+factor*value)%p
    return tuple(out)


def reconstruct_public(points:dict[int,tuple[int,...]], threshold:int, p:int) -> tuple[int,...]:
    if len(points)!=threshold:
        raise ValueError('Выберите ровно t опубликованных долей')
    sizes={len(v) for v in points.values()}
    if len(sizes)!=1: raise ValueError('Размерности долей различаются')
    return tuple(interpolate_coefficients({x:v[k] for x,v in points.items()},p)[0]
                 for k in range(next(iter(sizes))))


def alternative_curves(points:dict[int,int], threshold:int,p:int):
    if len(points)>=threshold or 0 in points:
        raise ValueError('Нужны < t опубликованных точек')
    return (interpolate_coefficients({0:0,**points},p),
            interpolate_coefficients({0:1,**points},p))


def horner(coefficients: tuple[int,...], x:int,p:int)->int:
    a=0
    for c in reversed(coefficients): a=(a*x+c)%p
    return a
