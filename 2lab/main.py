 
from __future__ import annotations

import matplotlib
matplotlib.use("TkAgg")

import math
from functools import reduce, wraps
from itertools import count, islice, chain
from typing import Iterable, Iterator, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


Point = Tuple[float, float]
Polygon = Tuple[Point, ...]



def polygon_edges(poly: Polygon):
    return zip(poly, poly[1:] + (poly[0],))


def distance(p1: Point, p2: Point) -> float:
    return math.dist(p1, p2)


def polygon_perimeter(poly: Polygon) -> float:
    return sum(distance(a, b) for a, b in polygon_edges(poly))


def polygon_area(poly: Polygon) -> float:
    s = 0

    for (x1, y1), (x2, y2) in polygon_edges(poly):
        s += x1 * y2 - x2 * y1

    return abs(s) / 2



#ВИЗУАЛИЗАЦИЯ


def visualize(polygons, title="Polygons"):
    polygons = list(polygons)

    fig, ax = plt.subplots(figsize=(10, 6))

    for poly in polygons:
        patch = MplPolygon(
            poly,
            closed=True,
            fill=False,
            linewidth=2
        )

        ax.add_patch(patch)

    ax.autoscale_view()

    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(title)

    plt.show(block=True)



#ГЕНЕРАТОРЫ


def gen_rectangle(
        width=2,
        height=1,
        step=3
) -> Iterator[Polygon]:

    for x in count(0, step):
        yield (
            (x, 0),
            (x + width, 0),
            (x + width, height),
            (x, height),
        )


def gen_triangle(
        side=2,
        step=3
) -> Iterator[Polygon]:

    h = side * math.sqrt(3) / 2

    for x in count(0, step):
        yield (
            (x, 0),
            (x + side / 2, h),
            (x + side, 0),
        )


def gen_hexagon(
        side=1,
        step=3
) -> Iterator[Polygon]:

    for x in count(0, step):

        yield tuple(
            (
                x + side * math.cos(math.radians(60 * i)),
                side * math.sin(math.radians(60 * i))
            )
            for i in range(6)
        )



#ТРАНСФОРМАЦИИ


def tr_translate(
        poly: Polygon,
        dx=0,
        dy=0
) -> Polygon:

    return tuple(
        (x + dx, y + dy)
        for x, y in poly
    )


def tr_rotate(
        poly: Polygon,
        angle=0,
        center=(0, 0)
) -> Polygon:

    rad = math.radians(angle)

    cx, cy = center

    result = []

    for x, y in poly:

        x -= cx
        y -= cy

        xr = x * math.cos(rad) - y * math.sin(rad)
        yr = x * math.sin(rad) + y * math.cos(rad)

        result.append((xr + cx, yr + cy))

    return tuple(result)


def tr_symmetry(
        poly: Polygon,
        axis="x"
) -> Polygon:

    if axis == "x":
        return tuple((x, -y) for x, y in poly)

    if axis == "y":
        return tuple((-x, y) for x, y in poly)

    if axis == "origin":
        return tuple((-x, -y) for x, y in poly)

    raise ValueError("Wrong axis")


def tr_homothety(
        poly: Polygon,
        k=1.0,
        center=(0, 0)
) -> Polygon:

    cx, cy = center

    return tuple(
        (
            cx + k * (x - cx),
            cy + k * (y - cy)
        )
        for x, y in poly
    )



#ФИЛЬТРЫ


def flt_convex_polygon(poly: Polygon) -> bool:
    signs = []

    pts = poly + (poly[0], poly[1])

    for i in range(len(poly)):

        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        x3, y3 = pts[i + 2]

        cross = (
            (x2 - x1) * (y3 - y2)
            - (y2 - y1) * (x3 - x2)
        )

        signs.append(cross > 0)

    return all(signs) or not any(signs)


def flt_square(
        poly: Polygon,
        max_area
) -> bool:

    return polygon_area(poly) < max_area


def flt_short_side(
        poly: Polygon,
        max_len
) -> bool:

    sides = [
        distance(a, b)
        for a, b in polygon_edges(poly)
    ]

    return min(sides) < max_len


def point_inside(poly: Polygon, point: Point):

    x, y = point

    inside = False

    for (x1, y1), (x2, y2) in polygon_edges(poly):

        cond = (
            ((y1 > y) != (y2 > y))
            and
            (
                x <
                (x2 - x1)
                * (y - y1)
                / (y2 - y1 + 1e-9)
                + x1
            )
        )

        if cond:
            inside = not inside

    return inside


def flt_point_inside(
        poly: Polygon,
        point: Point
):

    return point_inside(poly, point)



#ДЕКОРАТОРЫ


def decorator_filter(predicate):

    def outer(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            return filter(
                predicate,
                func(*args, **kwargs)
            )

        return wrapper

    return outer


def decorator_transform(
        transform,
        *targs,
        **tkwargs
):

    def outer(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            return map(
                lambda p:
                transform(
                    p,
                    *targs,
                    **tkwargs
                ),
                func(*args, **kwargs)
            )

        return wrapper

    return outer



#АГРЕГАЦИИ


def agr_origin_nearest(polygons):

    all_points = chain.from_iterable(polygons)

    return reduce(
        lambda acc, p:
        p
        if distance((0, 0), p)
           < distance((0, 0), acc)
        else acc,
        all_points
    )


def agr_max_side(polygons):

    edges = chain.from_iterable(
        map(polygon_edges, polygons)
    )

    return reduce(
        lambda acc, e:
        e
        if distance(*e)
           > distance(*acc)
        else acc,
        edges
    )


def agr_min_area(polygons):

    return reduce(
        lambda acc, p:
        p
        if polygon_area(p)
           < polygon_area(acc)
        else acc,
        polygons
    )


def agr_perimeter(polygons):

    return reduce(
        lambda acc, p:
        acc + polygon_perimeter(p),
        polygons,
        0
    )


def agr_area(polygons):

    return reduce(
        lambda acc, p:
        acc + polygon_area(p),
        polygons,
        0
    )



#ZIP POLYGONS


def zip_polygons(*iterables):

    for polys in zip(*iterables):

        yield tuple(
            chain.from_iterable(polys)
        )



#ОСНОВНОЙ ТЕСТ


if __name__ == "__main__":


    rectangles = list(
        islice(gen_rectangle(), 7)
    )

    triangles = list(
        islice(gen_triangle(), 7)
    )

    hexagons = list(
        islice(gen_hexagon(), 7)
    )

    visualize(rectangles, "Rectangles")
    visualize(triangles, "Triangles")
    visualize(hexagons, "Hexagons")


    base = list(
        islice(
            gen_rectangle(
                width=2,
                height=1
            ),
            7
        )
    )

    strip1 = list(
        map(
            lambda p:
            tr_rotate(p, 30),
            base
        )
    )

    strip2 = list(
        map(
            lambda p:
            tr_translate(p, 0, 3),
            strip1
        )
    )

    strip3 = list(
        map(
            lambda p:
            tr_translate(p, 0, 6),
            strip1
        )
    )

    visualize(
        chain(strip1, strip2, strip3),
        "Three strips"
    )


    base2 = list(
        islice(
            gen_rectangle(
                width=1.5,
                height=0.7
            ),
            8
        )
    )

    cross1 = list(
        map(
            lambda p:
            tr_rotate(p, 35),
            base2
        )
    )

    cross2 = list(
        map(
            lambda p:
            tr_translate(
                tr_rotate(p, -35),
                5,
                3
            ),
            base2
        )
    )

    visualize(
        chain(cross1, cross2),
        "Cross strips"
    )


    tri = list(
        islice(
            gen_triangle(),
            7
        )
    )

    tri2 = list(
        map(
            lambda p:
            tr_symmetry(p, "x"),
            tri
        )
    )

    visualize(
        chain(tri, tri2),
        "Symmetric triangles"
    )


    base_poly = (
        (0, 0),
        (1, 0),
        (1.5, 1),
        (0.5, 1.5),
    )

    scales = list(
        map(
            lambda k:
            tr_homothety(base_poly, k),
            [1, 2, 3, 4, 5]
        )
    )

    visualize(
        scales,
        "Homothety"
    )


    many = list(
        islice(
            gen_triangle(side=2),
            20
        )
    )

    filtered = list(
        filter(
            lambda p:
            flt_short_side(p, 2.5),
            many
        )
    )

    visualize(
        filtered[:6],
        "Filtered"
    )


    @decorator_transform(
        tr_translate,
        10,
        5
    )
    def decorated_rectangles():
        return list(
            islice(
                gen_rectangle(),
                5
            )
        )

    visualize(
        decorated_rectangles(),
        "Decorator transform"
    )

    @decorator_filter(
        lambda p:
        flt_square(p, 5)
    )
    def filtered_rectangles():
        return list(
            islice(
                gen_rectangle(
                    width=1,
                    height=1
                ),
                10
            )
        )

    visualize(
        filtered_rectangles(),
        "Decorator filter"
    )


    polys = list(
        islice(
            gen_rectangle(),
            5
        )
    )

    print()
    print("Nearest point:")
    print(agr_origin_nearest(polys))

    print()
    print("Max side:")
    print(agr_max_side(polys))

    print()
    print("Min area polygon:")
    print(agr_min_area(polys))

    print()
    print("Total perimeter:")
    print(agr_perimeter(polys))

    print()
    print("Total area:")
    print(agr_area(polys))


    zipped = list(
        islice(
            zip_polygons(
                gen_triangle(side=2),
                gen_rectangle(
                    width=1,
                    height=1
                )
            ),
            5
        )
    )

    visualize(
        zipped,
        "Zip polygons"
    )
 