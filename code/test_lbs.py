import numpy as np


def apply_lbs_skinning(vertices, bone_transforms, weights):
    # Start with zero influence
    transformed_vertices = np.zeros_like(vertices, dtype=float)

    # Convert vertices to homogeneous coordinates for matrix multiplication
    vertices_homogeneous = np.hstack([vertices, np.ones((vertices.shape[0], 1))])

    # Apply each bone's influence
    for i, bone_transform in enumerate(bone_transforms):
        for vertex_index, vertex in enumerate(vertices_homogeneous):
            # Apply transformation weighted by the bone's influence on the vertex
            transformed_vertices[vertex_index] += weights[i][vertex_index] * (bone_transform @ vertex)[:3]

    return transformed_vertices


if __name__ == "__main__":
    # Define vertices of the mesh
    vertices = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [0, -1, 0]
    ])

    # Define bone transformations (for simplicity, these are just translation matrices)
    bone_transforms = [
        np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),  # Move right
        np.array([[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # Move left
    ]

    # Vertex weights for each bone (how much each bone affects each vertex)
    weights = [
        [1, 0.5, 0, 0.5],
        [0, 0.5, 1, 0.5]
    ]

    # Apply the skinning
    transformed_vertices = apply_lbs_skinning(vertices, bone_transforms, weights)
    print(transformed_vertices)