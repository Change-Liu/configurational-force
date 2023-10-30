import paraview.simple as pvs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

def load_resampled_files(fileID):

    file_name = f"R{fileID}.vtk"
    print(f"Loading file: {file_name}")
    
    # Load the .vtk file
    dataset = pvs.OpenDataFile(file_name)

    return dataset

def extract_data_from_file(dataset):  
    # Update pipeline to ensure all properties are up-to-date
    pvs.UpdatePipeline()

    # Access the underlying VTK dataset
    vtk_data = pvs.servermanager.Fetch(dataset)

    # Get nodal coordinates
    vtk_points = vtk_data.GetPoints()
    num_points = vtk_points.GetNumberOfPoints()
    nodal_coordinates = [vtk_points.GetPoint(i) for i in range(num_points)]

    # Extract the element connectivity
    num_cells = vtk_data.GetNumberOfCells()
    cell_array = vtk_data.GetCells().GetData()

    connectivity = []
    idx = 0
    for i in range(num_cells):
        num_pts_in_cell = cell_array.GetValue(idx)
        idx += 1  # Move to the start of connectivity info for the cell
        cell_points = [cell_array.GetValue(idx + j) for j in range(num_pts_in_cell)]
        connectivity.append(cell_points)
        idx += num_pts_in_cell  # Move past the connectivity info to the next cell's data

    # Extract "v" field variable
    v_array = vtk_data.GetPointData().GetArray("v")
    v_values = [v_array.GetValue(i) for i in range(num_points)]

    # Convert to numpy arrays
    nodal_coordinates_np = np.array(nodal_coordinates)
    connectivity_np = np.array(connectivity, dtype=int)
    v_values_np = np.array(v_values)

    return nodal_coordinates_np, connectivity_np, v_values_np, num_points, num_cells

def quad_shape_functions(xi, eta):
    """Compute the shape functions for a 2D quadrilateral element."""
    N = np.array([
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta)
    ])
    return N

def quad_shape_function_derivatives(xi, eta):
    """Compute the derivatives of shape functions for a 2D quadrilateral element."""
    dN_dxi = np.array([
        -0.25 * (1 - eta),
         0.25 * (1 - eta),
         0.25 * (1 + eta),
        -0.25 * (1 + eta)
    ])
    dN_deta = np.array([
        -0.25 * (1 - xi),
        -0.25 * (1 + xi),
         0.25 * (1 + xi),
         0.25 * (1 - xi)
    ])

    dN= np.array([dN_dxi,dN_deta])
    return dN

def compute_jacobian(xi, eta, element_coords):
    """Compute the Jacobian matrix for a 2D quadrilateral element at the given xi and eta."""
    dN = quad_shape_function_derivatives(xi, eta)
    dN_dxi = dN[0]
    dN_deta = dN[1]

    jacobian = np.zeros((2, 2))
    for i in range(4): # assuming 4 nodes for the quad element
        jacobian[0, 0] += dN_dxi[i] * element_coords[i, 0]
        jacobian[0, 1] += dN_dxi[i] * element_coords[i, 1]
        jacobian[1, 0] += dN_deta[i] * element_coords[i, 0]
        jacobian[1, 1] += dN_deta[i] * element_coords[i, 1]

    return jacobian

def shape_function_derivatives_physical(xi, eta, element_coords):
    """
    Compute the derivatives of shape functions in the physical space
    for a 2D quadrilateral element at the given xi and eta.
    """
    dN = quad_shape_function_derivatives(xi, eta)
    dN_dxi = dN[0]
    dN_deta = dN[1]

    jacobian = compute_jacobian(xi, eta, element_coords)
    inv_jacobian = np.linalg.inv(jacobian)
    dN_dx = inv_jacobian[0, 0] * dN_dxi + inv_jacobian[0, 1] * dN_deta
    dN_dy = inv_jacobian[1, 0] * dN_dxi + inv_jacobian[1, 1] * dN_deta

    dNdx=np.array([dN_dx,dN_dy])
    return dNdx

def collect_shape_functions(element, nodal_coordinates):
    # define Gauss points
    element_coords = nodal_coordinates[element]
    g=np.zeros((2,2))
    gauss_points = np.array([[ -1 / np.sqrt(3), -1 / np.sqrt(3)],
                            [ 1 / np.sqrt(3), -1 / np.sqrt(3)],
                            [ 1 / np.sqrt(3),  1 / np.sqrt(3)],
                            [-1 / np.sqrt(3),  1 / np.sqrt(3)]])
    gauss_weights = np.array([1, 1, 1, 1])
    parametric_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    # calculate the shape functions and derivatives at the Gauss points
    N = np.zeros((4,4))
    dN = np.zeros((4,2,4))
    dNdx = np.zeros((4,2))

    for i in range(4):
                N[i]=quad_shape_functions(gauss_points[i,0],gauss_points[i,1]) * gauss_weights[i]
                dN[i]=quad_shape_function_derivatives(gauss_points[i,0],gauss_points[i,1]) * gauss_weights[i]
                jacobian=compute_jacobian(gauss_points[i,0],gauss_points[i,1],element_coords)
                # the gradient of the shape functions in the physical space
                dNdx_all=shape_function_derivatives_physical(parametric_coords[i,0],parametric_coords[i,1],element_coords)
                dNdx[i]=dNdx_all[:,i].T

    J = np.linalg.det(jacobian)

    return N, dN, dNdx, J

def configural_force(element, N, dN, dNdx, J, v_values, G_frac):
    # material parameters
    Gc = 20.2
    l = 1.0

    # Compute the fracture energy density
    v = v_values[element]

    for i in range(4):

        for ii in range(4):
            v_elem = np.dot(N[ii],v.T)
            dv_elem = np.dot(dN[ii],v.T)

            integrand = (Gc/(2*l))*(1-v_elem)**2+(Gc*l/2)*np.dot(dv_elem.T,dv_elem)
            psi_frac = integrand*np.eye(2)

            g_frac = np.matmul((psi_frac - Gc*l*np.matmul(dv_elem,dv_elem.T)),dNdx[i])
            # g_frac = (psi_frac - Gc*l*np.matmul(dv_elem,dv_elem.T))

            G_frac[element[i]] += g_frac * J

    return G_frac

def select_nodes_in_circle(nodal_coordinates, connectivity, radius, center):
    # Select nodes inside the circle and store node ID
    distances = np.linalg.norm(nodal_coordinates[:,0:2] - center, axis=1)
    selected_node_indices = np.where(distances <= radius)[0]
    
    selected_nodes = nodal_coordinates[selected_node_indices]
    
    return selected_nodes, selected_node_indices

def plot_mesh_and_selected_nodes(nodal_coordinates, connectivity, selected_nodes, cracked_nodes):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a list of polygons for each element
    polygons = [nodal_coordinates[element,0:2] for element in connectivity]
    
    # Use PolyCollection to plot all elements at once
    poly_collection = PolyCollection(polygons, edgecolors='lightgray', linewidths=0.1, facecolors='none')
    ax.add_collection(poly_collection)
    
    # Plot the selected nodes
    ax.scatter(selected_nodes[:, 0], selected_nodes[:, 1], c='b', s=0.1, marker='o', label='Selected Nodes')

    # Plot the cracked nodes
    ax.scatter(cracked_nodes[:, 0], cracked_nodes[:, 1], c='r', s=0.1, marker='o', label='Cracked Nodes')
    
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.legend()
    plt.savefig('mesh_with_selected_nodes.svg')

    return

def cracked_nodes(nodal_coordinates, v):
     # Select nodes with v>0.8
    cracked_node_indices = np.where(v > 0.8)[0]
    cracked_nodes = nodal_coordinates[cracked_node_indices]
    max_x_index = np.argmax(cracked_nodes[:, 0])
    node_with_max_x = cracked_nodes[max_x_index]
    crack_tip=node_with_max_x[0:2]

    return cracked_nodes, cracked_node_indices, crack_tip

# calculate the shape functions and derivatives at the Gauss points
dataset=load_resampled_files(0)
nodal_coordinates, connectivity, v, num_points, num_cells=extract_data_from_file(dataset)
N, dN, dNdx, J = collect_shape_functions(connectivity[0], nodal_coordinates)

# Calculate the configurational force for each time step
G_frac_time=np.zeros((4001,1))
crack_tip=np.zeros((4001,2))
for time_step in range(4001):
    dataset=load_resampled_files(time_step)
    nodal_coordinates, connectivity, v, num_points, num_cells=extract_data_from_file(dataset)

    # Select nodes inside the circle and store node ID
    cracked_node, cracked_node_indices, crack_tip[time_step] = cracked_nodes(nodal_coordinates, v)
    center = crack_tip[time_step]
    radius = 5
    selected_nodes, selected_node_ids = select_nodes_in_circle(nodal_coordinates, connectivity, radius, center)

    # Plot the mesh and selected nodes
    # print(center)
    # plot_mesh_and_selected_nodes(nodal_coordinates, connectivity, selected_nodes, cracked_nodes)

    G_frac=np.zeros((num_points,2))

    for element in range(num_cells):
        G_frac = configural_force(connectivity[element], N, dN, dNdx, J, v, G_frac)

    G_frac_time[time_step] = sum(G_frac[selected_node_ids,0])
    crack_tip_time = crack_tip[:,0]

# save the data
combined_data = np.column_stack((crack_tip_time, G_frac_time))

# Save to a .txt file
np.savetxt("output_file.txt", combined_data, fmt="%.5f", header="g_frac\tcrack_tip")

