import torch

def approx_cr(coordinates):
    """
    Approximate the square of cross-ratio along four ordered 2D points using 
    inner-product
    
    coordinates: list of 4 pytorch tensor of shape [2]
    """
    if len(coordinates) != 4:
        raise('Invalid argument: coordinates should be a list of 4 elements')
    AC = coordinates[2] - coordinates[0]
    BD = coordinates[3] - coordinates[1]
    BC = coordinates[2] - coordinates[1]
    AD = coordinates[3] - coordinates[0]
    return (AC.dot(AC) * BD.dot(BD)) / (BC.dot(BC) * AD.dot(AD))

def compute_cross_ratio_loss(output):
    # output = torch.randn(2, 33, 2) # Bx33x2
    points = output[-1, :-1] # remove the last point. It corresponds to the origin and is 
                        # irrelevant for cross_ratio_loss

    ## The 32 points are in this order:
    # First 8 points are the bounding box points
    # Rest are interplotion between Points ['X', 'Y']
    # A
    # B
    # C
    # D
    # E
    # F
    # G
    # H
    # ['A', 'B']
    # ['A', 'B']
    # ['A', 'C']
    # ['A', 'C']
    # ['A', 'E']
    # ['A', 'E']
    # ['B', 'D']
    # ['B', 'D']
    # ['B', 'F']
    # ['B', 'F']
    # ['C', 'D']
    # ['C', 'D']
    # ['C', 'G']
    # ['C', 'G']
    # ['D', 'H']
    # ['D', 'H']
    # ['E', 'F']
    # ['E', 'F']
    # ['E', 'G']
    # ['E', 'G']
    # ['F', 'H']
    # ['F', 'H']
    # ['G', 'H']
    # ['G', 'H']


    cross_ratio_loss = 0

    cross_ratio_loss += approx_cr([points[0],  points[8],  points[9], points[1]]) # ['A', 'B']
    cross_ratio_loss += approx_cr([points[0], points[10], points[11], points[2]]) # ['A', 'C']
    cross_ratio_loss += approx_cr([points[0], points[12], points[13], points[4]]) # ['A', 'E']
    cross_ratio_loss += approx_cr([points[1], points[14], points[15], points[3]]) # ['B', 'D']
    cross_ratio_loss += approx_cr([points[1], points[16], points[17], points[5]]) # ['B', 'F']
    cross_ratio_loss += approx_cr([points[2], points[18], points[19], points[3]]) # ['C', 'D']
    cross_ratio_loss += approx_cr([points[2], points[20], points[21], points[6]]) # ['C', 'G']
    cross_ratio_loss += approx_cr([points[3], points[22], points[23], points[7]]) # ['D', 'H']
    cross_ratio_loss += approx_cr([points[4], points[24], points[25], points[5]]) # ['E', 'F']
    cross_ratio_loss += approx_cr([points[4], points[26], points[27], points[6]]) # ['E', 'G']
    cross_ratio_loss += approx_cr([points[5], points[28], points[29], points[7]]) # ['F', 'H']
    cross_ratio_loss += approx_cr([points[6], points[30], points[31], points[7]]) # ['G', 'H']

    return cross_ratio_loss