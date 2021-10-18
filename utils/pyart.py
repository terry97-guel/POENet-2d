import torch
# def expm(vector):
#     if vector.shape[0] == 6:
#         return to_SE3(vector)
#     if vector.shape[0] == 4:
#         return to_SO3(vector)

def skew(p):
    device = p.device
    if len(p.size()) == 2:
        skew_p = torch.zeros(3,3,dtype=torch.float).to(device)
        p_temp = p

        skew_p[0,:] = torch.tensor([torch.tensor(0.0), -p_temp[2], p_temp[1]])
        skew_p[1,:] = torch.tensor([p_temp[2], torch.tensor(0.0), -p_temp[0]])
        skew_p[2,:] = torch.tensor([-p_temp[1], p_temp[0],torch.tensor(0.0)])

    elif len(p.size()) == 3:
        num_p = p.size()[0]
        skew_p = torch.zeros(num_p,3,3,dtype=torch.float).to(device)
    
        for i in range(num_p):
            p_temp = p[i]

            skew_p[i,0,:] = torch.tensor([torch.tensor(0.0), -p_temp[2], p_temp[1]])
            skew_p[i,1,:] = torch.tensor([p_temp[2], torch.tensor(0.0), -p_temp[0]])
            skew_p[i,2,:] = torch.tensor([-p_temp[1], p_temp[0],torch.tensor(0.0)])
    elif len(p.size()) == 1:
        skew_p = torch.tensor([
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]
        ]).to(device)
    else:
        raise(Exception("demension for p is invalid"))

    return skew_p

def rpy2r(rpy):
    device = rpy.deivce
    if len(rpy.size()) == 2:
        assert(rpy.size()[0] == 3)
        R = torch.zeros(3,3).to(device)
        num_rpy = 1

        rpy_temp = rpy
        r = rpy_temp[0]
        p = rpy_temp[1]
        y = rpy_temp[2]

        R[0,:] = torch.tensor([
            torch.cos(y)*torch.cos(p),
            -torch.sin(y)*torch.cos(r) + torch.cos(y)*torch.sin(p)*torch.sin(r),
            torch.sin(y)*torch.sin(r)+torch.cos(y)*torch.sin(p)*torch.cos(r)
            ])
        
        R[1,:] = torch.tensor([
            torch.sin(y)*torch.cos(p),
            torch.cos(y)*torch.cos(r) + torch.sin(y)*torch.sin(p)*torch.sin(r),
            -torch.cos(y)*torch.sin(r)+torch.sin(y)*torch.sin(p)*torch.cos(r)
            ])

        R[2,:] = torch.tensor([
            -torch.sin(p),
            torch.cos(p)*torch.sin(r),
            torch.cos(p)*torch.cos(r)
            ])
    elif len(rpy.size()) == 3:
        assert(rpy.size()[1] == 3)
        num_rpy = rpy.size()[0]
        R = torch.zeros(num_rpy,3,3)
        
        for i in range(num_rpy):
            rpy_temp = rpy[i]
            r = rpy_temp[0]
            p = rpy_temp[1]
            y = rpy_temp[2]

            R[i,0,:] = torch.tensor([
                torch.cos(y)*torch.cos(p),
                -torch.sin(y)*torch.cos(r) + torch.cos(y)*torch.sin(p)*torch.sin(r),
                torch.sin(y)*torch.sin(r)+torch.cos(y)*torch.sin(p)*torch.cos(r)
                ])
            
            R[i,1,:] = torch.tensor([
                torch.sin(y)*torch.cos(p),
                torch.cos(y)*torch.cos(r) + torch.sin(y)*torch.sin(p)*torch.sin(r),
                -torch.cos(y)*torch.sin(r)+torch.sin(y)*torch.sin(p)*torch.cos(r)
                ])

            R[i,2,:] = torch.tensor([
                -torch.sin(p),
                torch.cos(p)*torch.sin(r),
                torch.cos(p)*torch.cos(r)
                ])
    else:
        raise(Exception("demension for rpy is invalid"))
    
    return R

def pr2t(p,r):
    device = p.device
    assert(len(p.size()) == len(r.size()))
    if len(p.size()) == 2:
        assert(p.size()[0] == 3 & r.size()[0] == 3 & r.size()[0] == 3)
        T = torch.zeros(4,4,dtype=torch.float).to(device)

        T[0:3,0:3] = r
        T[0:3,3] =  p.squeeze()
        T[3,3] = 1
    elif len(p.size()) == 3:
        assert(p.size()[1] == 3 & r.size()[1] == 3 & r.size()[1] == 3)
        num_p = p.size()[0]
        T = torch.zeros(num_p,4,4,dtype=torch.float).to(device)

        for i in range(num_p):
            T[i,0:3,0:3] = r[i]
            T[i,0:3,3] =  p[i].squeeze()
            T[i,3,3] = 1
    else:
        raise(Exception("demension for position or orientation is invalid"))
    return T

def srodrigues(twist, q_value, verbose =False): #number of set of twist is one & number of q_value is n_joint
    eps = 1e-10
    device = twist.device
    batch_size = q_value.size(0)
    T = torch.zeros(batch_size,4,4,dtype=torch.float).to(device)

    #number of joint
    w = twist[:3]
    v = twist[3:]
    theta = w.norm(dim=0)

    if theta.item() < eps:
        theta = v.norm(dim=0)

    q_value = q_value * theta
    w = w/theta
    v = v/theta
    w_skew = skew(w)

    # print("q_value:", q_value.device)
    # print("w:", w.device)
    # print("(1-torch.cos(q_value):", (1-torch.cos(q_value)).device)
    # print("w_skew @ v:", (w_skew @ v).device)

    T[:,:3,:3] = rodrigues(w, q_value)
    T[:,:3,3] =  torch.outer(q_value,v) + \
        torch.outer((1-torch.cos(q_value)), w_skew @ v) + \
        torch.outer(q_value-torch.sin(q_value), w_skew@w_skew@v)
    T[:,3,3] = 1
    
    return T

def rodrigues(w,q,verbose = False):
    eps = 1e-10
    device = q.device
    batch_size = q.size()[0]

    if torch.norm(w) < eps:
        R = torch.tile(torch.eye(3),(batch_size,1,1)).to(device)
        return R
    if abs(torch.norm(w)-1) > eps:
        if verbose:
            print("Warning: [rodirgues] >> joint twist not normalized")

    theta = torch.norm(w)
    w = w/theta
    q = q*theta

    w_skew = skew(w)
    R = torch.tensordot(torch.ones_like(q).unsqueeze(0), torch.eye(3).unsqueeze(0).to(device), dims=([0],[0])) \
        + torch.tensordot(torch.sin(q).unsqueeze(0), w_skew.unsqueeze(0),dims = ([0],[0]))\
            + torch.tensordot( (1-torch.cos(q)).unsqueeze(0), (w_skew@w_skew).unsqueeze(0), dims =([0],[0]))
    return R
#%%