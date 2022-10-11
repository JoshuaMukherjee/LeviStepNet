import torch, math


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_board(N, z):  
    pitch=0.0105
    grid_vec=pitch*(torch.arange(-N/2+1, N/2, 1))
    x, y = torch.meshgrid(grid_vec,grid_vec,indexing="ij")
    trans_x=torch.reshape(x,(torch.numel(x),1))
    trans_y=torch.reshape(y,(torch.numel(y),1))
    trans_z=z*torch.ones((torch.numel(x),1))
    trans_pos=torch.cat((trans_x, trans_y, trans_z), axis=1)
    return trans_pos
  
def transducers():
  return torch.cat((create_board(17,.234/2),create_board(17,-.234/2)),axis=0).to(device)

def forward_model(points, transducers = transducers()):
    m=points.size()[0]
    n=transducers.size()[0]
    k=2*math.pi/0.00865
    radius=0.005
    transducers_x=torch.reshape(transducers[:,0],(n,1))
    transducers_y=torch.reshape(transducers[:,1],(n,1))
    transducers_z=torch.reshape(transducers[:,2],(n,1))
    points_x=torch.reshape(points[:,0],(m,1))
    points_y=torch.reshape(points[:,1],(m,1))
    points_z=torch.reshape(points[:,2],(m,1))
    

    distance=torch.sqrt((transducers_x.T-points_x)**2+(transducers_y.T-points_y)**2+(transducers_z.T-points_z)**2)
    planar_distance=torch.sqrt((transducers_x.T-points_x)**2+(transducers_y.T-points_y)**2)
    bessel_arg=k*radius*torch.divide(planar_distance,distance)
    directivity=1/2-bessel_arg**2/16+bessel_arg**4/384
    phase=torch.exp(1j*k*distance)
    trans_matrix=2*8.02*torch.multiply(torch.divide(phase,distance),directivity)
    return trans_matrix


def propagate(activations, points):
    A = forward_model(points)
    return A@activations


def permute_points(points,index,axis=0):
    if axis == 0:
        return points[index,:,:,:]
    if axis == 1:
        return points[:,index,:,:]
    if axis == 2:
        return points[:,:,index,:]
    if axis == 3:
        return points[:,:,:,index]