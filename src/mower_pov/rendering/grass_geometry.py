"""Clip grass ribbons with attribute interpolation at the new blade edge."""


def clip_blades(vertices, faces, colors, height):
    out_vertices, out_faces, out_colors = [], [], []
    indices = {}

    def insert(key, coordinate, color):
        if key not in indices:
            indices[key] = len(out_vertices)
            out_vertices.append(tuple(coordinate))
            out_colors.append(tuple(color))
        return indices[key]

    def is_below(index):
        return vertices[index][2] <= height

    def intersect(a, b):
        va, vb = vertices[a], vertices[b]
        fraction = (height - va[2]) / (vb[2] - va[2])
        if fraction <= 0.:
            return insert(a, va, colors[a])
        if fraction >= 1.:
            return insert(b, vb, colors[b])
        p = tuple(x + fraction * (y - x) for x, y in zip(va[:2], vb[:2])) + (height,)
        rgba = tuple(x + fraction * (y - x) for x, y in zip(colors[a], colors[b]))
        return insert(tuple(sorted((a, b))), p, rgba)

    for face in faces:
        polygon = []
        previous = face[-1]
        for current in face:
            inside = is_below(current)
            if inside != is_below(previous):
                polygon.append(intersect(previous, current))
            if inside:
                polygon.append(insert(current, vertices[current], colors[current]))
            previous = current
        polygon = tuple(dict.fromkeys(polygon))
        if len(polygon) >= 3:
            out_faces.append(polygon)
    return out_vertices, out_faces, out_colors
