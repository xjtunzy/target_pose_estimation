import numpy as np

def function_digital_case_with_max_error(ss):
    if ss < 10:
        return 1.767349 * ss**-1.102022
    else:
        return 1.436644 * ss**-1.004084

def maxlinedev_opt(points):
    front = points[0]
    back = points[-1]
    dev = []

    fx, fy = front
    bx, by = back

    deno = np.hypot(fy - by, fx - bx)
    if deno == 0:
        dev = [np.linalg.norm(np.array(p) - np.array(front)) for p in points]
    else:
        temp1 = (fy - by) / (fy * bx - by * fx) if (fy * bx - by * fx) != 0 else float('inf')
        temp2 = (fx - bx) / (fx * by - bx * fy) if (fx * by - bx * fy) != 0 else float('inf')
        deno = np.hypot(temp1, temp2)
        dev = [abs(p[0]*temp1 + p[1]*temp2 - 1) / deno for p in points]

    max_dev = max(dev)
    index = dev.index(max_dev)

    # 距离容差
    distances = [np.linalg.norm(np.array(p) - np.array(front)) for p in points]
    S_max = max(distances)
    del_phi_max = function_digital_case_with_max_error(S_max)
    del_tol_max = np.tan(del_phi_max) * S_max

    return {
        'max_dev': max_dev,
        'index': index,
        'D_temp': S_max,
        'del_tol_max': del_tol_max
    }

def simplify_rdp_with_indices(edge):
    """
    :param edge: list of (x, y) points
    :return: (simplified_points, simplified_indices)
    """
    if len(edge) < 3:
        return edge.copy(), list(range(len(edge)))

    simplified_points = [edge[0]]
    simplified_indices = [0]

    first = 0
    last = len(edge) - 1

    while first < last:
        segment = edge[first:last+1]
        result = maxlinedev_opt(segment)

        while result['max_dev'] > result['del_tol_max']:
            last = first + result['index']
            if first == last:
                break
            segment = edge[first:last+1]
            result = maxlinedev_opt(segment)

        if last == first:
            last = len(edge) - 1

        simplified_points.append(edge[last])
        simplified_indices.append(last)

        first = last
        last = len(edge) - 1

    return simplified_points, simplified_indices
