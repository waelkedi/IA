import math

def organize_data(data, whitelist=None):
    """
    "data" must follow this pattern: [((x, y), ["class1", ..]), ...].
    It will returns {"class1": [(x, y), ...]), ...}.
    You can also specify a filter to discard all classes that didn't match
    """
    organized_data = {}
    for position, classes in data:
        for class_ in classes:
            if whitelist is None or class_ in whitelist:
                position_list = organized_data.get(class_, [])
                position_list.append(position)
                organized_data[class_] = position_list
    return organized_data

def make_cluster(position_list, distance_max):
    """
    Groups the positions of a detected objects. We assume that the
    different positions of the same object is pretty close. So we make simple
    clustering according this assumption.
    A cluster is composed of positions that are separated by at most "distance_max".
    """
    clusters = []
    for position in position_list:
        cluster_found = False
        for cluster in clusters:
            if cluster.compute_distance(position) < distance_max:
                cluster.append(position)
                cluster_found = True
                break
        if not cluster_found:
            cluster = Cluster()
            cluster.append(position)
            clusters.append(cluster)
    return clusters

def find_object_position(data):
    found_object = {}
    for key in data.keys():
        clusters = make_cluster(data[key], 10)
        found_object[key] = [c.compute_means() for c in clusters]
    return found_object

def distance(p1, p2):
    delta_x = p1[0] - p2[0]
    delta_y = p1[1] - p2[1]
    return math.sqrt( delta_x**2 + delta_y**2)


class Cluster(list):

    def compute_means(self):
        sum_x, sum_y= 0, 0
        for position in self:
            sum_x += position[0]
            sum_y += position[1]
        return sum_x/len(self), sum_y/len(self)

    def compute_distance(self, position):
        mean = self.compute_means()
        return distance(mean, position)
